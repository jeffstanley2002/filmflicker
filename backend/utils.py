import pandas as pd


def safe_int(v):
    return int(v) if pd.notna(v) else None


def safe_float(v):
    return float(v) if pd.notna(v) else None


def safe_str(v):
    return str(v) if pd.notna(v) and v != "" else None
