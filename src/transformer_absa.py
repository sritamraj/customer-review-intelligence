from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DATA_DIR = Path("data/processed/semeval")

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

OUTPUT_DIR = Path("artifacts/absa_v3")

MODEL_NAME = "distilbert-base-uncased"

LABELS = [
    "positive",
    "negative",
    "neutral",
    "conflict",
]

LABEL2ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID2LABEL = {
    i: label
    for label, i in LABEL2ID.items()
}


def build_text(df):
    """
    Aspect-aware Transformer input.

    The aspect is explicitly provided together with the review:

        Review text [SEP] Aspect

    This forces the model to learn sentiment toward
    the requested aspect rather than only overall review sentiment.
    """

    return (
        df["review_text"].fillna("").astype(str)
        + " [SEP] "
        + df["aspect"].fillna("").astype(str)
    )


def load_dataframe(path):

    df = pd.read_csv(path)

    required_columns = {
        "review_text",
        "aspect",
        "sentiment",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["text"] = build_text(df)

    df["label"] = df["sentiment"].map(LABEL2ID)

    if df["label"].isna().any():
        bad_labels = sorted(
            df.loc[df["label"].isna(), "sentiment"]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown sentiment labels: {bad_labels}"
        )

    df["label"] = df["label"].astype(int)

    return df


def tokenize_dataset(dataset, tokenizer):

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    return dataset.map(
        tokenize,
        batched=True,
    )


def main():

    print("=" * 70)
    print("ABSA V3: DISTILBERT ASPECT-AWARE CLASSIFIER")
    print("=" * 70)

    for file in [
        TRAIN_FILE,
        VAL_FILE,
        TEST_FILE,
    ]:

        if not file.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file}"
            )

    train_df = load_dataframe(TRAIN_FILE)
    val_df = load_dataframe(VAL_FILE)
    test_df = load_dataframe(TEST_FILE)

    print("\nDataset sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    print("\nLabel distribution:")

    print(
        train_df["sentiment"]
        .value_counts()
        .to_string()
    )

    print("\nLoading tokenizer:")

    print(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = Dataset.from_pandas(
        train_df[
            ["text", "label"]
        ],
        preserve_index=False,
    )

    val_dataset = Dataset.from_pandas(
        val_df[
            ["text", "label"]
        ],
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_df[
            ["text", "label"]
        ],
        preserve_index=False,
    )

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
    )

    val_dataset = tokenize_dataset(
        val_dataset,
        tokenizer,
    )

    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
    )

    print("\nLoading Transformer model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABELS)),
        y=train_df["label"].values,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float,
    )

    print("\nClass weights:")

    for label, weight in zip(
        LABELS,
        class_weights,
    ):
        print(
            f"{label:10s}: {weight.item():.4f}"
        )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    def compute_metrics(eval_prediction):

        logits, labels = eval_prediction

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        macro_f1 = f1_score(
            labels,
            predictions,
            labels=np.arange(len(LABELS)),
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            labels,
            predictions,
            labels=np.arange(len(LABELS)),
            average="weighted",
            zero_division=0,
        )

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,

        num_train_epochs=4,

        weight_decay=0.01,

        logging_steps=25,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        report_to="none",

        save_total_limit=2,

        fp16=torch.cuda.is_available(),
    )

    class WeightedTrainer(Trainer):

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):

            labels = inputs.pop("labels")

            outputs = model(**inputs)

            logits = outputs.logits

            loss_function = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(logits.device)
            )

            loss = loss_function(
                logits,
                labels,
            )

            return (
                (loss, outputs)
                if return_outputs
                else loss
            )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n" + "=" * 70)
    print("TRAINING TRANSFORMER")
    print("=" * 70)

    trainer.train()

    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)

    val_results = trainer.evaluate(
        eval_dataset=val_dataset
    )

    print(val_results)

    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    test_output = trainer.predict(
        test_dataset
    )

    test_predictions = np.argmax(
        test_output.predictions,
        axis=-1,
    )

    y_test = test_df["label"].values

    print(
        classification_report(
            y_test,
            test_predictions,
            labels=np.arange(len(LABELS)),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        labels=np.arange(len(LABELS)),
        average="macro",
        zero_division=0,
    )

    test_weighted_f1 = f1_score(
        y_test,
        test_predictions,
        labels=np.arange(len(LABELS)),
        average="weighted",
        zero_division=0,
    )

    print(
        f"Test Macro-F1:    {test_macro_f1:.4f}"
    )

    print(
        f"Test Weighted-F1: {test_weighted_f1:.4f}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model_dir = (
        OUTPUT_DIR / "model"
    )

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    predictions_df = pd.DataFrame(
        {
            "review_text": test_df[
                "review_text"
            ],
            "aspect": test_df[
                "aspect"
            ],
            "true_sentiment": test_df[
                "sentiment"
            ],
            "predicted_sentiment": [
                ID2LABEL[int(pred)]
                for pred in test_predictions
            ],
        }
    )

    predictions_file = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False,
    )

    metrics_file = (
        OUTPUT_DIR
        / "metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ABSA V3 - DistilBERT\n"
        )

        f.write(
            f"Model: {MODEL_NAME}\n"
        )

        f.write(
            f"Test Macro-F1: {test_macro_f1:.4f}\n"
        )

        f.write(
            f"Test Weighted-F1: {test_weighted_f1:.4f}\n"
        )

    print("\n" + "=" * 70)
    print("V3 SAVED ARTIFACTS")
    print("=" * 70)

    print(
        f"Model:       {final_model_dir}"
    )

    print(
        f"Predictions: {predictions_file}"
    )

    print(
        f"Metrics:     {metrics_file}"
    )

    print("\nV3 training complete.")


if __name__ == "__main__":
    main()