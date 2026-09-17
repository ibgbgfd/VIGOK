import os
import sys
import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def generate_gaussian_target(height, width, pipes, max_radius_scale=100.0):
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius_map = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    
    for p in pipes:
        cx, cy, r = p["x"], p["y"], p["r"]
        int_cx, int_cy = int(round(cx)), int(round(cy))
        if int_cx < 0 or int_cx >= width or int_cy < 0 or int_cy >= height:
            continue
            
        sigma = max(1.5, min(5.0, r / 5.0))
        radius_window = int(max(3, 3 * sigma))
        x0 = max(0, int_cx - radius_window)
        x1 = min(width, int_cx + radius_window + 1)
        y0 = max(0, int_cy - radius_window)
        y1 = min(height, int_cy + radius_window + 1)
        
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        dist_sq = (xs - cx)**2 + (ys - cy)**2
        g = np.exp(-dist_sq / (2.0 * (sigma**2)))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)
        
        center_win = max(1, int(sigma))
        cx0 = max(0, int_cx - center_win)
        cx1 = min(width, int_cx + center_win + 1)
        cy0 = max(0, int_cy - center_win)
        cy1 = min(height, int_cy + center_win + 1)
        
        radius_map[cy0:cy1, cx0:cx1] = r / max_radius_scale
        mask[cy0:cy1, cx0:cx1] = 1.0
        
    return heatmap, radius_map, mask

def load_dataset_records(data_dir, split="train"):
    records = []
    
    # YOLO
    yolo_img = os.path.join(data_dir, "images", split)
    yolo_lbl = os.path.join(data_dir, "labels", split)
    if not os.path.exists(yolo_img):
        yolo_img = os.path.join(data_dir, split, "images")
        yolo_lbl = os.path.join(data_dir, split, "labels")

    if os.path.exists(yolo_img) and os.path.exists(yolo_lbl):
        for img_name in os.listdir(yolo_img):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            base = os.path.splitext(img_name)[0]
            lbl_file = os.path.join(yolo_lbl, base + ".txt")
            img_path = os.path.join(yolo_img, img_name)
            pipes = []
            if os.path.exists(lbl_file):
                try:
                    with Image.open(img_path) as im:
                        im_w, im_h = im.size
                except:
                    continue
                with open(lbl_file, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            _, cx_n, cy_n, w_n, h_n = map(float, parts[:5])
                            pipes.append({
                                "x": cx_n * im_w,
                                "y": cy_n * im_h,
                                "r": ((w_n * im_w) + (h_n * im_h)) / 4.0
                            })
            records.append({"image_path": img_path, "filename": img_name, "pipes": pipes})
        if records:
            return records, "yolo"

    # JSON
    json_path = os.path.join(data_dir, f"{split}_annotations.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(data_dir, split, f"{split}_annotations.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        img_folder = os.path.join(data_dir, split, "images")
        if not os.path.exists(img_folder):
            img_folder = os.path.join(data_dir, "images", split)
        if not os.path.exists(img_folder):
            img_folder = os.path.join(data_dir, split)
        for item in raw:
            fname = item.get("image_file") or item.get("file_name") or item.get("filename")
            ipath = os.path.join(img_folder, fname) if not os.path.isabs(fname) else fname
            records.append({"image_path": ipath, "filename": os.path.basename(fname), "pipes": item.get("pipes", [])})
        return records, "json"

    # CSV
    csv_path = os.path.join(data_dir, f"{split}.csv")
    if os.path.exists(csv_path):
        img_map = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fname = row.get("image") or row.get("filename")
                x = float(row.get("x") or row.get("cx") or 0)
                y = float(row.get("y") or row.get("cy") or 0)
                r = float(row.get("r") or row.get("radius") or (float(row.get("diameter", 0)) / 2.0))
                img_map.setdefault(fname, []).append({"x": x, "y": y, "r": r})
        img_folder = os.path.join(data_dir, split, "images")
        if not os.path.exists(img_folder):
            img_folder = os.path.join(data_dir, split)
        for fname, pipes in img_map.items():
            records.append({"image_path": os.path.join(img_folder, fname), "filename": fname, "pipes": pipes})
        return records, "csv"

    raise FileNotFoundError(f"Датасет '{split}' не найден в {data_dir} (поддерживаются YOLO, JSON, CSV)")

class PipeDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=256, augment=True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.records, fmt = load_dataset_records(data_dir, split)
        print(f"[{split}] Формат {fmt}, {len(self.records)} фото")
        
        self.cache = []
        for rec in self.records:
            if not os.path.exists(rec["image_path"]):
                continue
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
            except:
                continue
                
            orig_w, orig_h = img.size
            scale_x = self.img_size / float(orig_w)
            scale_y = self.img_size / float(orig_h)
            if (orig_w, orig_h) != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                
            scaled_pipes = []
            for p in rec["pipes"]:
                scaled_pipes.append({
                    "x": p["x"] * scale_x,
                    "y": p["y"] * scale_y,
                    "r": p["r"] * ((scale_x + scale_y) / 2.0)
                })
                
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
            hm, rad, msk = generate_gaussian_target(self.img_size, self.img_size, scaled_pipes)
            
            self.cache.append({
                "image": img_tensor,
                "heatmap": torch.from_numpy(hm).unsqueeze(0),
                "radius": torch.from_numpy(rad).unsqueeze(0),
                "mask": torch.from_numpy(msk).unsqueeze(0),
                "true_count": len(scaled_pipes),
                "filename": rec["filename"]
            })
            
    def __len__(self):
        return len(self.cache)
        
    def __getitem__(self, idx):
        item = self.cache[idx]
        if not self.augment:
            return item
            
        img, hm, rad, msk = item["image"], item["heatmap"], item["radius"], item["mask"]
        if torch.rand(1).item() > 0.5:
            img, hm, rad, msk = torch.flip(img, [2]), torch.flip(hm, [2]), torch.flip(rad, [2]), torch.flip(msk, [2])
        if torch.rand(1).item() > 0.5:
            img, hm, rad, msk = torch.flip(img, [1]), torch.flip(hm, [1]), torch.flip(rad, [1]), torch.flip(msk, [1])
            
        return {
            "image": img,
            "heatmap": hm,
            "radius": rad,
            "mask": msk,
            "true_count": item["true_count"],
            "filename": item["filename"]
        }
