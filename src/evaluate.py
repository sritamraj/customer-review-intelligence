import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SemEval ABSA sentiment predictions."
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to ABSA prediction CSV.",
    )

    args = parser.parse_args()

    predictions_path = Path(args.predictions)

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}"
        )

    df = pd.read_csv(predictions_path)

    required_columns = {
        "review_text",
        "aspect",
        "true_sentiment",
        "predicted_sentiment",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    y_true = df["true_sentiment"]
    y_pred = df["predicted_sentiment"]

    labels = [
        "positive",
        "negative",
        "neutral",
        "conflict",
    ]

    print("=" * 60)
    print("ABSA TEST SET EVALUATION")
    print("=" * 60)

    print(f"\nPrediction file: {predictions_path}")
    print(f"Number of examples: {len(df)}")

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            digits=4,
            zero_division=0,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    print("\n=== CONFUSION MATRIX ===")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[f"True: {label}" for label in labels],
        columns=[f"Pred: {label}" for label in labels],
    )

    print(confusion_df)

    print("\n=== MISCLASSIFIED EXAMPLES ===")

    errors = df[
        df["true_sentiment"] != df["predicted_sentiment"]
    ].copy()

    print(f"Number of errors: {len(errors)}")

    if len(errors) > 0:
        print("\nFirst 20 errors:\n")

        for i, (_, row) in enumerate(errors.head(20).iterrows(), 1):
            print(f"--- Error {i} ---")
            print(f"Aspect:     {row['aspect']}")
            print(f"True:       {row['true_sentiment']}")
            print(f"Predicted:  {row['predicted_sentiment']}")
            print(f"Review:     {row['review_text']}")
            print()

    # Save confusion matrix
    output_dir = predictions_path.parent

    confusion_path = output_dir / "confusion_matrix.csv"

    confusion_df.to_csv(confusion_path)

    # Save errors
    errors_path = output_dir / "misclassified_examples.csv"

    errors.to_csv(
        errors_path,
        index=False,
    )

    print("=" * 60)
    print("SAVED ARTIFACTS")
    print("=" * 60)

    print(f"Confusion matrix: {confusion_path}")
    print(f"Misclassified examples: {errors_path}")


if __name__ == "__main__":
    main()