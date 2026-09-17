import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataset import PipeDataset
from model import PipeNet, extract_peaks

def focal_mse_loss(pred_hm, target_hm):
    weights = 1.0 + 8.0 * target_hm
    return torch.mean(weights * ((pred_hm - target_hm) ** 2))

def masked_l1_loss(pred_rad, target_rad, mask):
    diff = torch.abs(pred_rad - target_rad) * mask
    return torch.sum(diff) / (torch.sum(mask) + 1e-6)

def train_model(data_dir="dataset", weights_dir="weights", epochs=20, batch_size=12, lr=1e-3, img_size=256, resume=None):
    os.makedirs(weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    train_dataset = PipeDataset(data_dir, split="train", img_size=img_size, augment=True)
    val_dataset = PipeDataset(data_dir, split="val", img_size=img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PipeNet(in_channels=3).to(device)
    if resume and os.path.exists(resume):
        print(f"Загрузка весов: {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    best_mae = float("inf")
    best_weights_path = os.path.join(weights_dir, "pipe_counter_best.pth")

    print(f"Старт обучения: {epochs} эпох, батч {batch_size}, размер {img_size}x{img_size}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            thm = batch["heatmap"].to(device)
            trad = batch["radius"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            phm, prad = model(imgs)
            loss = focal_mse_loss(phm, thm) + 1.2 * masked_l1_loss(prad, trad, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        count_diffs = []
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                thm = batch["heatmap"].to(device)
                trad = batch["radius"].to(device)
                mask = batch["mask"].to(device)
                true_cnts = batch["true_count"].numpy()

                phm, prad = model(imgs)
                val_losses.append((focal_mse_loss(phm, thm) + 1.2 * masked_l1_loss(prad, trad, mask)).item())

                detected = extract_peaks(phm, prad, threshold=0.32)
                for b_i, det in enumerate(detected):
                    count_diffs.append(abs(len(det) - true_cnts[b_i]))

        mae = float(np.mean(count_diffs))
        val_loss = float(np.mean(val_losses))
        train_loss = total_loss / len(train_loader)
        dt = time.time() - t0

        saved = ""
        if mae <= best_mae or epoch == 1:
            best_mae = mae
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "mae": best_mae, "img_size": img_size}, best_weights_path)
            saved = " *"

        print(f"[{epoch:02d}/{epochs:02d}] ({dt:.1f}s) loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | MAE: {mae:.2f}{saved}", flush=True)

    print(f"Готово. Лучший MAE: {best_mae:.2f}. Веса: {best_weights_path}")
    return best_weights_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset", help="Папка датасета")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--resume", default=None)
    p.add_argument("--weights_dir", default="weights")
    args = p.parse_args()

    train_model(args.data_dir, args.weights_dir, args.epochs, args.batch_size, args.lr, args.img_size, args.resume)
