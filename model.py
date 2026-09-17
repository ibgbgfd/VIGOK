import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PipeNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.enc1 = DepthwiseSeparableConv(24, 32)
        self.down1 = nn.MaxPool2d(2, 2) # H/2
        
        self.enc2 = DepthwiseSeparableConv(32, 64)
        self.down2 = nn.MaxPool2d(2, 2) # H/4
        
        self.enc3 = DepthwiseSeparableConv(64, 128)
        self.down3 = nn.MaxPool2d(2, 2) # H/8
        
        self.bottleneck = DepthwiseSeparableConv(128, 160)
        
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DepthwiseSeparableConv(160 + 128, 96)
        
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DepthwiseSeparableConv(96 + 64, 48)
        
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DepthwiseSeparableConv(48 + 32, 32)
        
        # Выходные головы
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.radius_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU()
        )
        
        # CenterNet/RetinaNet инициализация: сдвиг смещения последнего слоя
        # чтобы начальная вероятность фона была низкой (~0.1), предотвращая ложные пики
        final_conv = self.heatmap_head[2]
        if hasattr(final_conv, "bias") and final_conv.bias is not None:
            final_conv.bias.data.fill_(-2.19)

    def forward(self, x):
        s = self.stem(x)
        
        e1 = self.enc1(s)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        
        e3 = self.enc3(d2)
        d3 = self.down3(e3)
        
        b = self.bottleneck(d3)
        
        u3 = self.up3(b)
        c3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        heatmap = self.heatmap_head(c1)
        radius = self.radius_head(c1)
        return heatmap, radius

def extract_peaks(heatmap, radius_map, threshold=0.30, min_dist=5, max_radius_scale=100.0, max_peaks=120):
    """
    Быстрое извлечение центров труб через Non-Maximum Suppression 3x3 на тепловой карте.
    """
    # 2D NMS через max pooling
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (heatmap == hmax) & (heatmap > threshold)
    
    batch_results = []
    b_size = heatmap.shape[0]
    
    for b in range(b_size):
        b_keep = keep[b, 0]
        indices = torch.nonzero(b_keep, as_tuple=False)
        
        if len(indices) == 0:
            batch_results.append([])
            continue
            
        # Сбор уверенностей для отбора top-K
        confidences = heatmap[b, 0, indices[:, 0], indices[:, 1]]
        if len(indices) > max_peaks:
            topk_vals, topk_idx = torch.topk(confidences, k=max_peaks)
            indices = indices[topk_idx]
            confidences = topk_vals
            
        pipes = []
        for i in range(len(indices)):
            y, x = indices[i, 0].item(), indices[i, 1].item()
            conf = confidences[i].item()
            pred_r = radius_map[b, 0, y, x].item() * max_radius_scale
            if 3.0 <= pred_r <= 260.0:
                pipes.append({
                    "x": float(x),
                    "y": float(y),
                    "r": float(pred_r),
                    "confidence": float(conf)
                })
                
        pipes.sort(key=lambda p: p["confidence"], reverse=True)
        filtered = []
        for p in pipes:
            too_close = False
            for fp in filtered:
                dist = math.hypot(p["x"] - fp["x"], p["y"] - fp["y"])
                if dist < max(min_dist, min(p["r"], fp["r"]) * 0.45):
                    too_close = True
                    break
            if not too_close:
                filtered.append(p)
                
        batch_results.append(filtered)
        
    return batch_results
