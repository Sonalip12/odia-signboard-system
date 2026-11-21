"""
Classical image-restoration pipeline for degraded signboards.
Deliberately uses only OpenCV/NumPy (no pretrained deep models) so the whole
repository runs instantly on any machine with just `pip install -r
requirements.txt` — nothing to download.

Steps:
    1. Perspective correction  - finds the largest quadrilateral contour
       (the signboard) and warps it to a front-facing rectangle.
    2. Denoising               - fast non-local means denoising.
    3. Deblurring (sharpening) - unsharp mask.
    4. Contrast enhancement    - CLAHE on the luminance channel.
    5. Shadow / lighting correction - divide by a blurred (illumination)
       estimate.
"""
import cv2
import numpy as np


def correct_perspective(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    largest = max(contours, key=cv2.contourArea)
    h, w = gray.shape
    if cv2.contourArea(largest) < 0.2 * w * h:
        return img_bgr  # no clear board-sized region found; skip warp

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.float32)

    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left

    (tl, tr, br, bl) = ordered
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB), 10)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB), 10)

    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))
    return warped


def correct_lighting(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    illumination = cv2.GaussianBlur(gray, (0, 0), sigmaX=31)
    illumination = np.clip(illumination, 1, 255).astype(np.float32)
    norm = (gray.astype(np.float32) / illumination) * illumination.mean()
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = norm
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def denoise(img_bgr):
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, 7, 7, 7, 21)


def enhance_contrast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def sharpen(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 3)
    return cv2.addWeighted(img_bgr, 1.5, blur, -0.5, 0)


def restore(img_bgr, do_perspective=True):
    out = img_bgr.copy()
    if do_perspective:
        out = correct_perspective(out)
    out = denoise(out)
    out = correct_lighting(out)
    out = enhance_contrast(out)
    out = sharpen(out)
    return out
