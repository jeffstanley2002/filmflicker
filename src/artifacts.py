"""Atomic model artifact publication and release manifests."""
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.recommenders.base import MODELS_DIR

MANIFEST_PATH = MODELS_DIR / "artifact_manifest.json"


def atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(*, generation_id: str | None = None, config: dict | None = None) -> dict:
    previous = {}
    if MANIFEST_PATH.exists():
        try:
            previous = json.loads(MANIFEST_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
    generation_id = generation_id or previous.get("generation_id") or str(uuid.uuid4())
    names = [
        "popularity.csv",
        "content_tfidf_vectorizer.joblib",
        "content_tfidf_matrix.npz",
        "content_movie_ids.npy",
        "collaborative_svd.joblib",
        "clustering.joblib",
        "neural_weights.npz",
        "metrics.json",
    ]
    checksums = {name: sha256(MODELS_DIR / name) for name in names if (MODELS_DIR / name).exists()}
    catalog_manifest = MODELS_DIR.parent / "data" / "processed" / "catalog_manifest.json"
    manifest = {
        "schema_version": 1,
        "generation_id": generation_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "checksums": checksums,
        "catalog_manifest_sha256": sha256(catalog_manifest) if catalog_manifest.exists() else None,
        "config": config or previous.get("config") or {},
    }
    atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2) + "\n")
    return manifest
