from pathlib import Path
import pandas as pd
from .utils import clean_text

REQUIRED = {"review", "complaint", "complaint_category"}

def load_reviews(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["review"] = df["review"].map(clean_text)
    df = df[df["review"].str.len() > 0]
    df["complaint"] = df["complaint"].astype(int)
    df["complaint_category"] = df["complaint_category"].fillna("none").astype(str)
    return df.reset_index(drop=True)
