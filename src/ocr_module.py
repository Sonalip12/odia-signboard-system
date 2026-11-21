"""
OCR + script/language handling for Odia signboards, using Tesseract's
'ori' (Odia) trained data, with a word-level confidence estimate used to
flag uncertain recognitions. Falls back to English OCR if Odia recognition
returns nothing (handles mixed Odia/English signboards).
"""
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def _preprocess_for_ocr(img_bgr):
    # The restoration module already denoises, corrects lighting, boosts
    # contrast, and sharpens. Here we only upscale (Tesseract does much
    # better on larger text) and convert to grayscale — aggressive binary
    # thresholding on an already-restored image tends to destroy strokes
    # of the Odia script, which has many conjuncts and curves.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    return gray


def _run_tesseract(proc, lang):
    data = pytesseract.image_to_data(
        proc, lang=lang, config="--psm 7", output_type=Output.DICT
    )
    words, confs = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf = float(conf)
        if text and conf >= 0:
            words.append(text)
            confs.append(conf)
    full_text = " ".join(words).strip()
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return full_text, avg_conf, list(zip(words, confs))


def _has_odia_glyphs(text):
    # Odia Unicode block: U+0B00 - U+0B7F
    return any("\u0B00" <= ch <= "\u0B7F" for ch in text)


def recognize(img_bgr, lang="ori"):
    """Odia-first recognition (this is a regional Odia-signboard system),
    with an English fallback for mixed-script signboards where no Odia
    glyphs are detected at all."""
    proc = _preprocess_for_ocr(img_bgr)

    text_ori, conf_ori, wc_ori = _run_tesseract(proc, "ori")

    if _has_odia_glyphs(text_ori) and conf_ori > 0:
        full_text, avg_conf, word_confs = text_ori, conf_ori, wc_ori
    else:
        # likely an English-only or non-Odia portion of a mixed signboard
        full_text, avg_conf, word_confs = _run_tesseract(proc, "eng")

    return {
        "recognized_text": full_text,
        "confidence": round(avg_conf, 1),
        "word_confidences": word_confs,
    }


def is_low_confidence(result, threshold=45.0):
    return (not result["recognized_text"]) or result["confidence"] < threshold
