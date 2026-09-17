import os
import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from predict import load_pipe_model, detect_pipes, draw_pipe_annotations

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def process_image(path, model, device, img_size, threshold, min_dia, max_dia, output=None, quiet=False):
    try:
        orig_img, pipes, stats = detect_pipes(
            str(path), model, device,
            threshold=threshold,
            min_radius=min_dia / 2.0,
            max_radius=max_dia / 2.0,
            target_size=img_size
        )
    except Exception as e:
        print(f"Error {path}: {e}", file=sys.stderr)
        return 0

    cnt = stats["total"]
    if output:
        draw_pipe_annotations(orig_img, pipes, stats).save(output, quality=95)

    if quiet:
        print(cnt)
    else:
        print(f"{Path(path).name}: {cnt}")
    return cnt

def main():
    p = argparse.ArgumentParser(description="Подсчет количества труб на фото")
    p.add_argument("input", help="Путь к фото или папке")
    p.add_argument("--weights", default="weights/pipe_counter_best.pth", help="Веса модели")
    p.add_argument("--threshold", type=float, default=0.32, help="Порог уверенности (0.1 - 0.9)")
    p.add_argument("--min-diameter", type=float, default=6.0, help="Мин. диаметр в px")
    p.add_argument("--max-diameter", type=float, default=500.0, help="Макс. диаметр в px")
    p.add_argument("--output", "-o", default=None, help="Сохранить фото с разметкой")
    p.add_argument("--quiet", "-q", action="store_true", help="Выводить только число")
    args = p.parse_args()

    target = Path(args.input)
    if not target.exists():
        print(f"Путь не найден: {args.input}", file=sys.stderr)
        sys.exit(1)

    model, device, img_size = load_pipe_model(args.weights)

    if target.is_file():
        process_image(target, model, device, img_size, args.threshold, args.min_diameter, args.max_diameter, args.output, args.quiet)
    elif target.is_dir():
        files = sorted([f for f in target.iterdir() if f.suffix.lower() in EXTS])
        total = 0
        for f in files:
            out_file = os.path.join(args.output, f"det_{f.name}") if args.output else None
            if args.output:
                os.makedirs(args.output, exist_ok=True)
            total += process_image(f, model, device, img_size, args.threshold, args.min_diameter, args.max_diameter, out_file, args.quiet)
        if not args.quiet:
            print(f"Всего: {total}")

if __name__ == "__main__":
    main()
