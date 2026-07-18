import pandas as pd


def safe_int(value) -> int | None:
    return int(value) if pd.notna(value) else None


def safe_float(value) -> float | None:
    return float(value) if pd.notna(value) else None


def safe_str(value) -> str | None:
    return str(value) if pd.notna(value) and value != "" else None
