"""
Runs the full pipeline over the generated dataset and reports accuracy,
broken down by degradation level, plus how well the confidence flag
correlates with actual correctness (useful for showing the
"uncertainty communication" aspect of the project).

Usage:
    python3 evaluate.py --dataset_dir ../dataset
"""
import os
import csv
import argparse
import cv2

from pipeline import run_pipeline


def evaluate(dataset_dir):
    labels_path = os.path.join(dataset_dir, "labels.csv")
    images_dir = os.path.join(dataset_dir, "images")

    with open(labels_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        img_path = os.path.join(images_dir, row["filename"])
        out = run_pipeline(img_path)
        correct = out["english_translation"].startswith(row["english_gt"])
        results.append({
            **row,
            "predicted_translation": out["english_translation"],
            "ocr_confidence": out["ocr_confidence"],
            "low_confidence_flag": out["low_confidence_flag"],
            "correct": correct,
        })

    # overall + per-level accuracy
    by_level = {}
    for r in results:
        lvl = r["degradation_level"]
        by_level.setdefault(lvl, []).append(r["correct"])

    print("\n=== Evaluation Report ===")
    total_correct = sum(r["correct"] for r in results)
    print(f"Overall accuracy: {total_correct}/{len(results)} "
          f"({100 * total_correct / len(results):.1f}%)")
    for lvl, vals in by_level.items():
        acc = 100 * sum(vals) / len(vals)
        print(f"  degradation={lvl:8s} accuracy: {sum(vals)}/{len(vals)} ({acc:.1f}%)")

    # confidence-flag usefulness: of the flagged-low-confidence results,
    # what fraction were actually wrong? (high = flag is trustworthy)
    flagged = [r for r in results if r["low_confidence_flag"]]
    if flagged:
        flagged_wrong = sum(1 for r in flagged if not r["correct"])
        print(f"\nOf {len(flagged)} low-confidence-flagged results, "
              f"{flagged_wrong} were actually wrong "
              f"({100 * flagged_wrong / len(flagged):.1f}% flag precision).")

    out_csv = os.path.join(dataset_dir, "evaluation_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-image results written -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "dataset"))
    args = parser.parse_args()
    evaluate(args.dataset_dir)
