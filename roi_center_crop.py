#!/usr/bin/env python3
"""
Center-crop ROI for frames to reduce noise and cost.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm

IN_DIR = "dataset_v2/frames"
OUT_DIR = "dataset_v2/frames_roi"
CROP_RATIO = 0.6


def center_crop(im: Image.Image, ratio: float) -> Image.Image:
    w, h = im.size
    cw, ch = int(w * ratio), int(h * ratio)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return im.crop((x0, y0, x0 + cw, y0 + ch))


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    paths = list(Path(IN_DIR).rglob("*.png"))
    for p in tqdm(paths, desc="cropping"):
        rel = p.relative_to(IN_DIR)
        outp = Path(OUT_DIR) / rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(p).convert("RGB")
        cropped = center_crop(im, CROP_RATIO)
        cropped.save(outp)

    print("[DONE] ROI frames saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
