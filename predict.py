
import os
import argparse
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import PipeNet, extract_peaks

def load_pipe_model(weights_path="weights/pipe_counter_best.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = PipeNet(in_channels=3).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        img_size = checkpoint.get("img_size", 256)
    else:
        model.load_state_dict(checkpoint)
        img_size = 256
    model.eval()
    return model, device, img_size

def detect_pipes(
    image_input,
    model,
    device,
    threshold=0.32,
    min_radius=3.0,
    max_radius=300.0,
    target_size=256
):
    """
    Принимает PIL.Image или путь к файлу.
    Возвращает:
      orig_img: исходное изображение PIL
      pipes: список обнаруженных труб {'x', 'y', 'r', 'confidence', 'category'}
      stats: статистика по размерам
    """
    if isinstance(image_input, str):
        orig_img = Image.open(image_input).convert("RGB")
    else:
        orig_img = image_input.convert("RGB")
        
    orig_w, orig_h = orig_img.size
    
    # Ресайз для подачи в нейросеть
    resized_img = orig_img.resize((target_size, target_size), Image.BILINEAR)
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmap, radius_map = model(img_tensor)
        raw_detections = extract_peaks(heatmap, radius_map, threshold=threshold)[0]
        
    # Масштабируем координаты обратно к оригинальному разрешению
    scale_x = orig_w / float(target_size)
    scale_y = orig_h / float(target_size)
    avg_scale = (scale_x + scale_y) / 2.0
    
    pipes = []
    for d in raw_detections:
        real_x = d["x"] * scale_x
        real_y = d["y"] * scale_y
        real_r = d["r"] * avg_scale
        
        if real_r < min_radius or real_r > max_radius:
            continue
            
        pipes.append({
            "x": round(real_x, 1),
            "y": round(real_y, 1),
            "r": round(real_r, 1),
            "diameter": round(real_r * 2.0, 1),
            "confidence": round(d["confidence"], 3)
        })
        
    # Сортировка сверху вниз для удобной нумерации
    pipes.sort(key=lambda p: (p["y"], p["x"]))
    
    # Категоризация по размерам (мелкие, средние, крупные)
    if pipes:
        radii = [p["r"] for p in pipes]
        r_min, r_max = min(radii), max(radii)
        r_range = max(1.0, r_max - r_min)
        
        small_thresh = r_min + r_range * 0.33
        large_thresh = r_min + r_range * 0.66
        
        small_cnt = 0
        med_cnt = 0
        large_cnt = 0
        
        for i, p in enumerate(pipes, 1):
            p["id"] = i
            if p["r"] < small_thresh:
                p["category"] = "small"
                small_cnt += 1
            elif p["r"] < large_thresh:
                p["category"] = "medium"
                med_cnt += 1
            else:
                p["category"] = "large"
                large_cnt += 1
                
        stats = {
            "total": len(pipes),
            "small": small_cnt,
            "medium": med_cnt,
            "large": large_cnt,
            "min_diameter": round(r_min * 2.0, 1),
            "max_diameter": round(r_max * 2.0, 1),
            "avg_diameter": round(np.mean([p["diameter"] for p in pipes]), 1)
        }
    else:
        stats = {
            "total": 0, "small": 0, "medium": 0, "large": 0,
            "min_diameter": 0.0, "max_diameter": 0.0, "avg_diameter": 0.0
        }
        
    return orig_img, pipes, stats

def draw_pipe_annotations(image, pipes, stats, show_labels=True):
    """
    Отрисовывает контуры труб, центры, номера и красивый инфо-блок со статистикой.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    
    # Цвета категорий (RGBA)
    colors = {
        "small": ((0, 220, 255, 230), (0, 220, 255, 40)),       # Циан
        "medium": ((50, 255, 120, 230), (50, 255, 120, 40)),    # Салатовый
        "large": ((255, 165, 0, 230), (255, 165, 0, 40))        # Оранжевый
    }
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        header_font = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font = ImageFont.load_default()
        header_font = font

    # 1. Отрисовка всех найденных кругов труб
    for p in pipes:
        cx, cy, r = p["x"], p["y"], p["r"]
        cat = p.get("category", "medium")
        stroke_col, fill_col = colors.get(cat, colors["medium"])
        
        # Полупрозрачная подсветка круга
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill_col, outline=stroke_col, width=max(2, int(r * 0.08)))
        
        # Центр трубы
        cross_size = max(2, int(r * 0.15))
        draw.line([cx - cross_size, cy, cx + cross_size, cy], fill=stroke_col, width=2)
        draw.line([cx, cy - cross_size, cx, cy + cross_size], fill=stroke_col, width=2)
        
        # Номер трубы в центре
        if show_labels and r > 8:
            num_str = str(p["id"])
            # Подложка под текст
            tw = len(num_str) * 7 + 6
            th = 14
            draw.rectangle([cx - tw//2, cy - th//2, cx + tw//2, cy + th//2], fill=(15, 20, 30, 200), outline=stroke_col)
            draw.text((cx - tw//2 + 3, cy - th//2), num_str, fill=(255, 255, 255, 255), font=font)

    # 2. Инфо-плашка в углу (HUD)
    hud_w = 260
    hud_h = 135
    margin = 15
    draw.rounded_rectangle(
        [margin, margin, margin + hud_w, margin + hud_h],
        radius=10,
        fill=(15, 23, 42, 220), # Темно-синий slate
        outline=(59, 130, 246, 255), # Яркий синий бордюр
        width=2
    )
    
    draw.text((margin + 12, margin + 10), f"ВСЕГО ТРУБ: {stats['total']}", fill=(255, 255, 255), font=header_font)
    
    # Статистика по категориям
    draw.text((margin + 12, margin + 40), f"• Мелкие (D < {stats.get('min_diameter', 0) + (stats.get('max_diameter', 0)-stats.get('min_diameter', 0))*0.33:.1f}px): {stats['small']}", fill=(0, 220, 255), font=font)
    draw.text((margin + 12, margin + 60), f"• Средние: {stats['medium']}", fill=(50, 255, 120), font=font)
    draw.text((margin + 12, margin + 80), f"• Крупные: {stats['large']}", fill=(255, 165, 0), font=font)
    draw.text((margin + 12, margin + 105), f"Диаметры: от {stats['min_diameter']} до {stats['max_diameter']} px (ср. {stats['avg_diameter']})", fill=(200, 210, 230), font=font)
    
    return annotated

def process_file(image_path, weights_path="weights/pipe_counter_best.pth", output_path="output_detected.jpg", threshold=0.32):
    print(f"Обработка изображения: {image_path}")
    model, device, img_size = load_pipe_model(weights_path)
    orig_img, pipes, stats = detect_pipes(image_path, model, device, threshold=threshold, target_size=img_size)
    annotated = draw_pipe_annotations(orig_img, pipes, stats)
    annotated.save(output_path, quality=95)
    
    print("\n" + "="*50)
    print(f"РЕЗУЛЬТАТЫ ПОДСЧЕТА ТРУБ:")
    print(f"-> Всего обнаружено труб: {stats['total']}")
    print(f"   - Мелких:   {stats['small']}")
    print(f"   - Средних:  {stats['medium']}")
    print(f"   - Крупных:  {stats['large']}")
    print(f"   - Мин. диаметр: {stats['min_diameter']} px")
    print(f"   - Макс. диаметр: {stats['max_diameter']} px")
    print(f"-> Размеченное фото сохранено в: {output_path}")
    print("="*50)
    return stats, pipes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подсчет труб на фото с помощью обученной нейросети")
    parser.add_argument("image", type=str, help="Путь к фотографии с трубами")
    parser.add_argument("--weights", type=str, default="weights/pipe_counter_best.pth", help="Путь к файлу весов модели")
    parser.add_argument("--output", type=str, default="output_detected.jpg", help="Путь для сохранения размеченного фото")
    parser.add_argument("--threshold", type=float, default=0.35, help="Порог уверенности детекции (0.1 - 0.9)")
    args = parser.parse_args()
    
    process_file(args.image, args.weights, args.output, args.threshold)
