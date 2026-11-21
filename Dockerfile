FROM python:3.11-slim

# System dependencies: Tesseract OCR engine, the Odia ('ori') language
# pack, and the Noto Sans Oriya font (used to generate the synthetic
# dataset). All installed here so the end user needs nothing but Docker.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-ori fonts-noto-core \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x run_demo.sh

CMD ["./run_demo.sh"]
