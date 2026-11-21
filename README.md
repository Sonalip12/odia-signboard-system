# AI-Based Odia Regional System for Signboard Image Restoration, Recognition, and Translation

End-to-end pipeline: **degraded Odia signboard image → perspective correction
& restoration → Odia OCR (with confidence estimation) → English translation**.

Text-to-speech is intentionally **not implemented** (omitted per project scope).

## Why a synthetic dataset?

No public, labeled dataset of real degraded Odia signboards exists. We
generate one: common Odia signboard phrases (Hospital, Police Station,
School, Bus Stand, Temple, etc.) are rendered in the Odia script and then
degraded with perspective tilt, blur, sensor noise, lighting/shadow
gradients, local fading patches, and JPEG compression artifacts —
mirroring what real-world signboards look like. This gives exact ground
truth to measure the pipeline's accuracy. See `src/vocabulary.py` for the
full phrase list and `src/generate_dataset.py` for the generator.

## Run it — three options, no manual setup needed

### Option A : Google Colab notebook
Open `notebooks/demo.ipynb` in Google Colab ("Open in Colab" works directly
from GitHub once this repo is pushed). The first cell installs everything
automatically in Colab's cloud environment — **nothing is installed on the
reviewer's own computer**. Just click "Run all".

OPEN THIS: https://colab.research.google.com/github/Sonalip12/odia-signboard-system/blob/main/notebooks/demo.ipynb

### Option B: Docker (one command, fully self-contained)
```bash
docker build -t odia-signboard .
docker run --rm odia-signboard
```
This builds an image with Tesseract, the Odia language pack, the Odia
font, and all Python packages baked in, generates the dataset, and prints
the evaluation report. No local Python/OCR installation required at all.

### Option C: Local machine (one setup script)
```bash
bash setup.sh      # installs tesseract-ocr, Odia language pack, Odia font, Python deps
bash run_demo.sh   # generates dataset, runs sample + full evaluation
```

## Repository structure
```
odia-signboard-system/
├── src/
│   ├── vocabulary.py        # Odia signboard phrase list + offline translation dictionary
│   ├── generate_dataset.py  # synthetic degraded-signboard dataset generator
│   ├── restoration.py       # perspective correction, denoising, lighting fix, CLAHE, sharpening
│   ├── ocr_module.py        # Tesseract-based Odia OCR + confidence scoring
│   ├── pipeline.py          # ties restoration -> OCR -> translation together
│   └── evaluate.py          # runs the pipeline over the dataset, reports accuracy
├── dataset/                 # generated on first run (images/ + labels.csv)
├── notebooks/demo.ipynb     # Colab-ready walkthrough notebook
├── outputs/                 # sample restored images land here
├── requirements.txt
├── setup.sh
├── run_demo.sh
└── Dockerfile
```

## Pipeline details

| Stage | Method |
|---|---|
| Perspective correction | Contour-based quadrilateral detection + homography warp (OpenCV) |
| Restoration | Non-local means denoising, illumination normalization, CLAHE contrast, unsharp-mask sharpening |
| OCR | Tesseract with the Odia (`ori`) trained-data model; falls back to English for mixed-script text |
| Confidence estimation | Mean word-level confidence from Tesseract; results below threshold are flagged as uncertain |
| Translation | Offline dictionary lookup (closed vocabulary of common signboard phrases) — no internet/API needed |

## Evaluation

Run `src/evaluate.py` (or `run_demo.sh`) to get a report like:
```
Overall accuracy: 56/120 (46.7%)
  degradation=low      accuracy: 25/40 (62.5%)
  degradation=medium   accuracy: 18/40 (45.0%)
  degradation=high     accuracy: 13/40 (32.5%)

Of 65 low-confidence-flagged results, 55 were actually wrong (84.6% flag precision).
```
Accuracy drops as degradation increases (as expected), and the confidence
flag is a reasonably reliable indicator of when the result shouldn't be
trusted — directly demonstrating the "communicate uncertainty" goal from
the project proposal.

## Team
Sonali Pradhan (Roll No. 26), Aditi Panda (Roll No. 64) — 5th Sem, CSE, Section C
