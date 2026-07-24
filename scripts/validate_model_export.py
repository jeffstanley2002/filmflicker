"""Deep validation of a complete, internally consistent serving bundle."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.artifacts import MANIFEST_PATH, sha256

MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
REQUIRED_MODEL_FILES = (
    "popularity.csv",
    "content_tfidf_vectorizer.joblib",
    "content_tfidf_matrix.npz",
    "content_movie_ids.npy",
    "collaborative_svd.joblib",
    "clustering.joblib",
    "neural_weights.npz",
    "metrics.json",
)


def _finite(name: str, value, failures: list[str]) -> None:
    if not np.isfinite(np.asarray(value)).all():
        failures.append(f"{name} contains NaN or infinite values")


def validate() -> list[str]:
    failures = []
    for name in REQUIRED_MODEL_FILES:
        path = MODELS / name
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"Missing or empty model artifact: {path}")
    if failures:
        return failures

    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Missing or invalid artifact manifest: {exc}"]
    if not manifest.get("generation_id"):
        failures.append("Artifact manifest has no generation_id")
    for name in REQUIRED_MODEL_FILES:
        expected = manifest.get("checksums", {}).get(name)
        if not expected:
            failures.append(f"Manifest has no checksum for {name}")
        elif sha256(MODELS / name) != expected:
            failures.append(f"Checksum mismatch for {name}")

    try:
        popularity = pd.read_csv(MODELS / "popularity.csv")
        required_columns = {"movieId", "count", "mean", "weighted_score"}
        if not required_columns.issubset(popularity.columns):
            failures.append("Popularity artifact is missing serving columns")
        _finite("popularity scores", popularity[["count", "mean", "weighted_score"]], failures)

        content_ids = np.load(MODELS / "content_movie_ids.npy")
        content_matrix = sparse.load_npz(MODELS / "content_tfidf_matrix.npz")
        vectorizer = joblib.load(MODELS / "content_tfidf_vectorizer.joblib")
        if content_matrix.shape[0] != len(content_ids):
            failures.append("Content matrix row count does not match movie IDs")
        if content_matrix.shape[1] != len(vectorizer.vocabulary_):
            failures.append("Content matrix width does not match vectorizer vocabulary")
        _finite("content matrix", content_matrix.data, failures)

        collaborative = joblib.load(MODELS / "collaborative_svd.joblib")
        serving_keys = {"item_factors", "movie_ids", "movie_idx", "movie_bias", "global_mean"}
        if not serving_keys.issubset(collaborative):
            failures.append("Collaborative artifact is missing serving keys")
        if {"svd", "user_factors", "user_ids", "user_means"} & set(collaborative):
            failures.append("Collaborative artifact still contains training-only arrays")
        if collaborative["item_factors"].shape[0] != len(collaborative["movie_ids"]):
            failures.append("Collaborative factor rows do not match movie IDs")
        if len(collaborative["movie_bias"]) != len(collaborative["movie_ids"]):
            failures.append("Collaborative biases do not match movie IDs")
        _finite("collaborative factors", collaborative["item_factors"], failures)
        _finite("collaborative biases", collaborative["movie_bias"], failures)

        clusters = joblib.load(MODELS / "clustering.joblib")
        if len(clusters["movie_ids"]) != len(clusters["labels"]):
            failures.append("Cluster labels do not match movie IDs")

        with np.load(MODELS / "neural_weights.npz") as neural:
            n_movies = len(neural["movie_ids"])
            if any(len(neural[name]) != n_movies for name in ("movie_emb", "movie_bias", "genre_matrix")):
                failures.append("Embedding arrays do not match movie IDs")
            mode = str(np.asarray(neural.get("training_mode", "missing")).item())
            if mode != "two_tower_neural":
                failures.append(
                    "Neural artifact must be independently trained two_tower_neural; "
                    f"found {mode!r}"
                )
            if neural["Wi"].shape[0] != neural["movie_emb"].shape[1] + neural["genre_matrix"].shape[1]:
                failures.append("Neural item tower input width does not match movie embeddings plus genres")
            if neural["Wi"].shape[1] != len(neural["bi"]):
                failures.append("Neural item tower output width does not match bias")
            for name in neural.files:
                if np.issubdtype(neural[name].dtype, np.number):
                    _finite(f"neural {name}", neural[name], failures)
    except Exception as exc:
        failures.append(f"Artifact loading failed: {type(exc).__name__}: {exc}")

    try:
        metrics = json.loads((MODELS / "metrics.json").read_text())
        for key in ("rating_prediction", "top_n", "k", "n_eval_users"):
            if key not in metrics:
                failures.append(f"metrics.json missing key: {key}")
        neural_evidence = metrics.get("neural_independence", {})
        if neural_evidence.get("training_mode") != "two_tower_neural":
            failures.append("metrics.json does not document two_tower_neural independence evidence")
        if metrics.get("n_eval_users", 0) < 100:
            failures.append("Evaluation sample is too small (minimum 100 users)")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"Invalid metrics.json: {exc}")

    catalog_path = PROCESSED / "catalog_manifest.json"
    if not catalog_path.exists():
        failures.append("No catalog manifest found; run scripts/build_catalog.py")
    elif manifest.get("catalog_manifest_sha256") != sha256(catalog_path):
        failures.append("Catalog manifest checksum does not match the trained bundle")
    return failures


def main() -> None:
    failures = validate()
    if failures:
        print("Model export validation failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Model export validation passed.")


if __name__ == "__main__":
    main()
