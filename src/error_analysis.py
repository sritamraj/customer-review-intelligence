import argparse
from pathlib import Path

import pandas as pd


def classify_error(true_label, predicted_label):
    if true_label == predicted_label:
        return "correct"

    if true_label == "positive" and predicted_label == "negative":
        return "positive_to_negative"

    if true_label == "negative" and predicted_label == "positive":
        return "negative_to_positive"

    if true_label == "neutral":
        return "neutral_confusion"

    if true_label == "conflict":
        return "conflict_missed"

    if predicted_label == "neutral":
        return "predicted_neutral"

    if predicted_label == "conflict":
        return "predicted_conflict"

    return "other_error"


def main():

    parser = argparse.ArgumentParser(
        description="Analyze ABSA model errors."
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to ABSA prediction CSV.",
    )

    args = parser.parse_args()

    predictions_file = Path(args.predictions)

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_file}"
        )

    output_dir = predictions_file.parent

    output_file = (
        output_dir / "error_analysis.csv"
    )

    confusion_file = (
        output_dir / "confusion_pairs.csv"
    )

    df = pd.read_csv(predictions_file)

    required_columns = {
        "review_text",
        "aspect",
        "true_sentiment",
        "predicted_sentiment",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["correct"] = (
        df["true_sentiment"]
        == df["predicted_sentiment"]
    )

    df["error_type"] = df.apply(
        lambda row: classify_error(
            row["true_sentiment"],
            row["predicted_sentiment"],
        ),
        axis=1,
    )

    total = len(df)

    correct = int(
        df["correct"].sum()
    )

    errors = total - correct

    print("=" * 70)
    print("ABSA ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\nPrediction file: {predictions_file}"
    )

    print(
        f"Total examples: {total}"
    )

    print(
        f"Correct: {correct}"
    )

    print(
        f"Errors: {errors}"
    )

    print(
        f"Accuracy: {correct / total:.4f}"
    )

    print("\n=== ERROR TYPE COUNTS ===")

    print(
        df["error_type"]
        .value_counts()
        .to_string()
    )

    print("\n=== TRUE SENTIMENT COUNTS ===")

    print(
        df["true_sentiment"]
        .value_counts()
        .to_string()
    )

    print("\n=== PREDICTED SENTIMENT COUNTS ===")

    print(
        df["predicted_sentiment"]
        .value_counts()
        .to_string()
    )

    print("\n=== CONFUSION PAIRS ===")

    errors_df = df[
        df["true_sentiment"]
        != df["predicted_sentiment"]
    ]

    confusion_pairs = (
        errors_df
        .groupby(
            [
                "true_sentiment",
                "predicted_sentiment",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    if len(confusion_pairs) > 0:
        print(
            confusion_pairs.to_string(
                index=False
            )
        )
    else:
        print("No errors found.")

    print("\n=== ASPECT-LEVEL PERFORMANCE ===")

    aspect_stats = (
        df.groupby("aspect")
        .agg(
            examples=("correct", "size"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )

    aspect_stats["errors"] = (
        aspect_stats["examples"]
        - aspect_stats["correct"]
    )

    aspect_stats["accuracy"] = (
        aspect_stats["correct"]
        / aspect_stats["examples"]
    )

    aspect_stats = aspect_stats.sort_values(
        [
            "examples",
            "accuracy",
        ],
        ascending=[
            False,
            True,
        ],
    )

    print(
        aspect_stats.head(30).to_string(
            index=False
        )
    )

    print("\n=== CONFLICT ERRORS ===")

    conflict_errors = df[
        (
            df["true_sentiment"]
            == "conflict"
        )
        &
        (
            df["predicted_sentiment"]
            != "conflict"
        )
    ]

    print(
        f"Conflict examples missed: "
        f"{len(conflict_errors)}"
    )

    for i, (_, row) in enumerate(
        conflict_errors.iterrows(),
        1,
    ):

        print(
            f"\n--- Conflict Error {i} ---"
        )

        print(
            f"Aspect: {row['aspect']}"
        )

        print(
            f"True: {row['true_sentiment']}"
        )

        print(
            f"Predicted: "
            f"{row['predicted_sentiment']}"
        )

        print(
            f"Review: {row['review_text']}"
        )

    print("\n=== FIRST 20 GENERAL ERRORS ===")

    for i, (_, row) in enumerate(
        errors_df.head(20).iterrows(),
        1,
    ):

        print(
            f"\n--- Error {i} ---"
        )

        print(
            f"Aspect: {row['aspect']}"
        )

        print(
            f"True: {row['true_sentiment']}"
        )

        print(
            f"Predicted: "
            f"{row['predicted_sentiment']}"
        )

        print(
            f"Type: {row['error_type']}"
        )

        print(
            f"Review: {row['review_text']}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )

    confusion_pairs.to_csv(
        confusion_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SAVED ARTIFACTS")
    print("=" * 70)

    print(
        f"Error analysis: {output_file}"
    )

    print(
        f"Confusion pairs: {confusion_file}"
    )


if __name__ == "__main__":
    main()