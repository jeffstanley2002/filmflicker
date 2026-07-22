"""Run the full train/evaluate/export validation workflow.

This intentionally shells out to the existing scripts so their logs remain
clear and each stage can still be run independently during experimentation.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    python = sys.executable
    run([python, "scripts/train_models.py"])
    run([python, "scripts/evaluate_models.py"])
    run([python, "scripts/validate_model_export.py"])


if __name__ == "__main__":
    main()
