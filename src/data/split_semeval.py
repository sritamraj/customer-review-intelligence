"""Create leakage-safe train/validation/test splits for SemEval ABSA.

All aspect annotations from the same source sentence are assigned to one
split. If sentence_id is unavailable, identical review_text values are used
as the grouping key. The split is approximately 70/15/15 using
StratifiedGroupKFold; exact sizes can vary because groups cannot be divided.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data/processed/semeval_laptop_absa.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/semeval"
LABELS = {"positive", "negative", "neutral", "conflict"}
RANDOM_STATE = 42
N_SPLITS = 20


def create_grouped_splits(df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = RANDOM_STATE):
    required = {"review_text", "aspect", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Input contains missing review_text, aspect, or sentiment values.")

    data = df.copy().reset_index(drop=True)
    data["sentiment"] = data["sentiment"].astype(str).str.strip().str.lower()
    unexpected = set(data["sentiment"]) - LABELS
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels: {sorted(unexpected)}")

    if "sentence_id" in data.columns and data["sentence_id"].notna().all():
        groups = data["sentence_id"].astype(str)
    else:
        # Fallback is conservative: duplicate sentence text cannot leak across splits.
        groups = data["review_text"].astype(str)

    unique_groups = groups.nunique()
    if unique_groups < n_splits:
        raise ValueError(f"Need at least {n_splits} unique sentence groups; found {unique_groups}.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_for_row = pd.Series(index=data.index, dtype="int64")
    dummy_x = data["review_text"].astype(str)
    for fold, (_, held_out) in enumerate(splitter.split(dummy_x, data["sentiment"], groups)):
        fold_for_row.iloc[held_out] = fold

    # 14/20 train, 3/20 validation, 3/20 test: approximately 70/15/15.
    test_mask = fold_for_row.isin([0, 1, 2])
    val_mask = fold_for_row.isin([3, 4, 5])
    train_mask = ~(test_mask | val_mask)
    splits = {
        "train": data.loc[train_mask].copy(),
        "validation": data.loc[val_mask].copy(),
        "test": data.loc[test_mask].copy(),
    }

    # Explicitly verify group isolation.
    group_sets = {name: set(groups.loc[part.index]) for name, part in splits.items()}
    names = list(group_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise RuntimeError(f"Group leakage between {left} and {right}: {len(overlap)} groups")

    return splits


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prepared SemEval CSV not found: {INPUT_FILE}\n"
            "First run: python -m src.data.prepare_semeval"
        )
    df = pd.read_csv(INPUT_FILE)
    splits = create_grouped_splits(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        path = OUTPUT_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"{name:10s}: {len(part):4d} rows -> {path.relative_to(PROJECT_ROOT)}")
        print(part["sentiment"].value_counts().sort_index().to_string())
    print("\nGroup leakage check: PASSED")
    print("Note: split sizes are approximate because complete sentences are kept together.")


if __name__ == "__main__":
    main()
