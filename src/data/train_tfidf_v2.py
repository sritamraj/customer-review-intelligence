from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


DATA_DIR = Path("data/processed/semeval")

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

OUTPUT_DIR = Path("artifacts/absa_v2")
MODEL_FILE = OUTPUT_DIR / "tfidf_word_char_logistic_regression.joblib"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"


LABELS = [
    "positive",
    "negative",
    "neutral",
    "conflict",
]


def build_input(df):
    """
    Create an aspect-aware input representation.

    Example:
        review text + [ASPECT] + aspect term

    This allows the classifier to use the review context
    together with the specific aspect being evaluated.
    """

    return (
        df["review_text"].fillna("").astype(str)
        + " [ASPECT] "
        + df["aspect"].fillna("").astype(str)
    )


def build_model():

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )

    features = FeatureUnion(
        [
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ]
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    model = Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )

    return model


def evaluate(model, x, y, split_name):

    predictions = model.predict(x)

    macro_f1 = f1_score(
        y,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

    print(
        classification_report(
            y,
            predictions,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print(f"{split_name} Macro-F1:    {macro_f1:.4f}")
    print(f"{split_name} Weighted-F1: {weighted_f1:.4f}")

    return predictions, macro_f1, weighted_f1


def main():

    print("=" * 70)
    print("ABSA V2: WORD + CHARACTER TF-IDF")
    print("=" * 70)

    for file in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file}"
            )

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    required_columns = {
        "review_text",
        "aspect",
        "sentiment",
    }

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: "
                f"{sorted(missing)}"
            )

    print("\nDataset sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    x_train = build_input(train_df)
    y_train = train_df["sentiment"]

    x_val = build_input(val_df)
    y_val = val_df["sentiment"]

    x_test = build_input(test_df)
    y_test = test_df["sentiment"]

    print("\nBuilding V2 model...")

    model = build_model()

    print("\nTraining...")
    model.fit(x_train, y_train)

    print("\nEvaluating validation set...")

    val_predictions, val_macro_f1, val_weighted_f1 = evaluate(
        model,
        x_val,
        y_val,
        "Validation",
    )

    print("\nEvaluating test set...")

    test_predictions, test_macro_f1, test_weighted_f1 = evaluate(
        model,
        x_test,
        y_test,
        "Test",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    predictions_df = pd.DataFrame(
        {
            "review_text": test_df["review_text"],
            "aspect": test_df["aspect"],
            "true_sentiment": y_test,
            "predicted_sentiment": test_predictions,
        }
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("V2 SAVED ARTIFACTS")
    print("=" * 70)

    print(f"Model:       {MODEL_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")

    print("\n" + "=" * 70)
    print("V2 SUMMARY")
    print("=" * 70)

    print(f"Validation Macro-F1: {val_macro_f1:.4f}")
    print(f"Test Macro-F1:       {test_macro_f1:.4f}")
    print(f"Test Weighted-F1:    {test_weighted_f1:.4f}")

    print("\nV2 training complete.")


if __name__ == "__main__":
    main()