#!/usr/bin/env bash
# One-shot setup: installs Tesseract OCR + the Odia ('ori') language pack,
# the Noto Sans Oriya font (needed to generate the synthetic dataset), and
# all Python dependencies. Run this once, then use run_demo.sh.
set -e

echo ">> Installing system packages (tesseract OCR + Odia language data + Odia font)..."
if command -v apt-get >/dev/null; then
    apt-get update -qq
    apt-get install -y tesseract-ocr tesseract-ocr-ori fonts-noto-core
else
    echo "This script expects a Debian/Ubuntu system with apt-get."
    echo "Please install: tesseract-ocr, tesseract-ocr-ori (Odia language data), and the Noto Sans Oriya font manually."
fi

echo ">> Installing Python dependencies..."
pip install -r requirements.txt

echo ">> Setup complete."
