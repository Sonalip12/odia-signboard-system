"""
Generates a synthetic dataset of degraded Odia signboard images.

Why synthetic? There is no public, labeled dataset of real degraded Odia
signboards. We instead render clean Odia text (common signboard phrases)
onto realistic sign-like backgrounds, then apply the same kinds of
degradations real signboards suffer from: perspective tilt, blur, noise,
low light / shadow, and compression artifacts. This gives exact ground
truth (the original clean text + translation) to evaluate the pipeline.

Run:
    python3 src/generate_dataset.py --n_per_class 5
"""
import os
import csv
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from vocabulary import SIGNBOARD_VOCAB

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansOriya-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansOriyaUI-Regular.ttf",
]

BOARD_COLORS = [
    ((20, 90, 40), (255, 255, 255)),    # green board, white text
    ((160, 20, 20), (255, 255, 255)),   # red board, white text
    ((10, 40, 120), (255, 255, 255)),   # blue board, white text
    ((245, 245, 230), (20, 20, 20)),    # cream board, dark text
    ((255, 215, 0), (10, 10, 10)),      # yellow board, dark text
]


def _get_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise FileNotFoundError(
        "Noto Sans Oriya font not found. Install with: "
        "apt-get install -y fonts-noto-core"
    )


def _render_clean_signboard(text, width=640, height=220):
    bg_color, text_color = random.choice(BOARD_COLORS)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # fit font size to width
    size = 60
    font = _get_font(size)
    while True:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw < width * 0.85 and th < height * 0.6:
            break
        size -= 2
        font = _get_font(size)
        if size < 10:
            break

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((width - tw) / 2 - bbox[0], (height - th) / 2 - bbox[1])
    draw.text(pos, text, fill=text_color, font=font)

    # simple border like a real signboard frame
    draw.rectangle([4, 4, width - 5, height - 5], outline=text_color, width=4)
    return img


def _random_perspective(img):
    w, h = img.size
    arr = np.array(img)
    import cv2
    jitter = int(min(w, h) * random.uniform(0.03, 0.14))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, jitter), random.randint(0, jitter)],
        [w - random.randint(0, jitter), random.randint(0, jitter)],
        [w - random.randint(0, jitter), h - random.randint(0, jitter)],
        [random.randint(0, jitter), h - random.randint(0, jitter)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(warped)


def _degrade(img, level="medium"):
    import cv2
    arr = np.array(img).astype(np.float32)

    # 1. lighting / shadow gradient
    h, w = arr.shape[:2]
    gradient = np.tile(np.linspace(
        random.uniform(0.5, 0.9), random.uniform(0.9, 1.1), w
    ), (h, 1))
    if random.random() < 0.5:
        gradient = gradient[:, ::-1]
    arr *= gradient[:, :, None]

    # 2. gaussian blur (simulate out-of-focus / motion)
    sigma = {"low": 0.8, "medium": 1.6, "high": 2.6}[level]
    arr = cv2.GaussianBlur(arr, (0, 0), sigma)

    # 3. sensor / compression noise
    noise_std = {"low": 4, "medium": 9, "high": 16}[level]
    arr += np.random.normal(0, noise_std, arr.shape)

    # 4. random local "stain/fade" patches (missing/faded characters)
    n_patches = {"low": 0, "medium": 1, "high": 2}[level]
    for _ in range(n_patches):
        px, py = random.randint(0, w - 1), random.randint(0, h - 1)
        pr = random.randint(int(0.05 * w), int(0.15 * w))
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - px) ** 2 + (yy - py) ** 2 <= pr ** 2
        fade = random.uniform(0.3, 0.7)
        arr[mask] = arr[mask] * fade + 255 * (1 - fade)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)

    # 5. JPEG-style compression artifact
    import io
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=random.randint(35, 70))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def generate(n_per_class=5, out_dir="dataset", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    idx = 0
    levels = ["low", "medium", "high"]
    for odia_text, english in SIGNBOARD_VOCAB:
        for i in range(n_per_class):
            level = levels[i % len(levels)]
            clean = _render_clean_signboard(odia_text)
            warped = _random_perspective(clean)
            degraded = _degrade(warped, level=level)

            fname = f"img_{idx:04d}.jpg"
            degraded.save(os.path.join(img_dir, fname), quality=80)
            rows.append({
                "filename": fname,
                "odia_text_gt": odia_text,
                "english_gt": english,
                "degradation_level": level,
            })
            idx += 1

    csv_path = os.path.join(out_dir, "labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "odia_text_gt", "english_gt", "degradation_level"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {idx} images -> {img_dir}")
    print(f"Labels written -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "dataset"))
    args = parser.parse_args()
    generate(args.n_per_class, args.out_dir)
