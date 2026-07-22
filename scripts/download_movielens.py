"""Download a MovieLens dataset by profile.

The app can still run on ml-latest-small, but serious retraining should use
MovieLens 32M or full latest when you are ready for the longer runtime.
"""
import argparse
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"

DATASETS = {
    "latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "latest-full": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    url = DATASETS[args.dataset]
    folder_name = Path(url).stem
    target = DATA_DIR / folder_name
    if target.exists() and not args.force:
        print(f"{target} already exists; use --force to redownload.")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / Path(url).name
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=(10, 180)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", "0"))
        downloaded = 0
        last_reported_mb = -1
        with zip_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                downloaded_mb = downloaded // (1024 * 1024)
                if downloaded_mb != last_reported_mb and downloaded_mb % 25 == 0:
                    last_reported_mb = downloaded_mb
                    if total:
                        pct = downloaded / total * 100
                        print(f"Downloaded {downloaded_mb:,} MB ({pct:.1f}%)", flush=True)
                    else:
                        print(f"Downloaded {downloaded_mb:,} MB", flush=True)

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {target}")


if __name__ == "__main__":
    main()
