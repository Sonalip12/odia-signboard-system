"""
End-to-end pipeline:
    degraded signboard image
        -> perspective correction + restoration
        -> Odia OCR (with confidence)
        -> dictionary-based Odia -> English translation
        -> final structured result (with low-confidence flag)

Text-to-speech is intentionally NOT implemented (omitted per project scope).

Usage:
    python3 pipeline.py --image ../dataset/images/img_0002.jpg
    python3 pipeline.py --image ../dataset/images/img_0002.jpg --save_restored out.jpg
"""
import argparse
import cv2

from restoration import restore
from ocr_module import recognize, is_low_confidence
from vocabulary import translate


def run_pipeline(image_path, save_restored=None, do_perspective=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    restored = restore(img, do_perspective=do_perspective)
    if save_restored:
        cv2.imwrite(save_restored, restored)

    ocr_result = recognize(restored)
    recognized_text = ocr_result["recognized_text"]
    confidence = ocr_result["confidence"]
    low_conf = is_low_confidence(ocr_result)

    translation = translate(recognized_text) if recognized_text else \
        "Translation unavailable (no text recognized)"

    return {
        "image_path": image_path,
        "recognized_odia_text": recognized_text,
        "ocr_confidence": confidence,
        "low_confidence_flag": low_conf,
        "english_translation": translation,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--save_restored", default=None)
    parser.add_argument("--no_perspective", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        args.image, args.save_restored, do_perspective=not args.no_perspective
    )
    print("\n--- Signboard Recognition Result ---")
    for k, v in result.items():
        print(f"{k}: {v}")
    if result["low_confidence_flag"]:
        print("\n[!] Low-confidence recognition — result may be unreliable.")
