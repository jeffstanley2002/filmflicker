"""Validate exported model artifacts and catalog readiness before deploy."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

REQUIRED_MODEL_FILES = [
    "popularity.csv",
    "content_tfidf_vectorizer.joblib",
    "content_tfidf_matrix.npz",
    "content_movie_ids.npy",
    "collaborative_svd.joblib",
    "clustering.joblib",
    "neural_weights.npz",
    "metrics.json",
]


def main():
    failures = []
    for name in REQUIRED_MODEL_FILES:
        path = MODELS / name
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"Missing or empty model artifact: {path}")

    metrics_path = MODELS / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        for key in ["rating_prediction", "top_n", "k"]:
            if key not in metrics:
                failures.append(f"metrics.json missing key: {key}")
        top_n = metrics.get("top_n", {})
        if top_n and not any(
            any(name.startswith("precision_at_") and isinstance(value, (int, float)) and value > 0 for name, value in vals.items())
            for vals in top_n.values()
        ):
            failures.append("metrics.json has no positive ranking precision values.")

    manifest_path = PROCESSED / "catalog_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        max_year = manifest.get("max_year")
        if max_year and max_year < 2020:
            failures.append(f"Catalog max year is {max_year}; use MovieLens 32M or a fresher free catalog before launch.")
    else:
        failures.append("No data/processed/catalog_manifest.json found. Run scripts/build_catalog.py.")

    if failures:
        print("Model export validation failed:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("Model export validation passed.")


if __name__ == "__main__":
    main()
