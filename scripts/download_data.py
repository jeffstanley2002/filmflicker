"""Download and unpack the MovieLens ml-latest-small dataset.

Run once: python scripts/download_data.py
Writes CSVs into data/ml-latest-small/, which get committed to the repo so
the deployed app never needs network access to load core data.
"""
import io
import zipfile
from pathlib import Path

import requests

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    target = DATA_DIR / "ml-latest-small"
    if (target / "movies.csv").exists():
        print(f"Dataset already present at {target}, skipping download.")
        return

    print(f"Downloading {DATA_URL} ...")
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()

    print("Unzipping ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_DIR)

    print(f"Done. Data available at {target}")


if __name__ == "__main__":
    main()
