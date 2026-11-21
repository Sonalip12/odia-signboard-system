#!/usr/bin/env bash
# Generates the synthetic dataset (if not already present) and runs the
# full pipeline + evaluation report. Run ./setup.sh first (or use Docker).
set -e
cd "$(dirname "$0")/src"

if [ ! -f "../dataset/labels.csv" ]; then
    echo ">> Generating synthetic Odia signboard dataset..."
    python3 generate_dataset.py --n_per_class 6
fi

echo ">> Running pipeline on one sample image..."
python3 pipeline.py --image ../dataset/images/img_0000.jpg --save_restored ../outputs/sample_restored.jpg

echo ">> Running full evaluation over the dataset..."
python3 evaluate.py

echo ">> Done. See dataset/evaluation_results.csv for per-image results."
