import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_PATH = "data/processed/semeval/train.csv"
VAL_PATH = "data/processed/semeval/validation.csv"
TEST_PATH = "data/processed/semeval/test.csv"

OUTPUT_DIR = "artifacts/absa_v4/model"
PREDICTION_PATH = "artifacts/absa_v4/test_predictions.csv"
METRICS_PATH = "artifacts/absa_v4/metrics.txt"

MAX_LENGTH = 256

LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01

# Focal Loss parameter
FOCAL_GAMMA = 2.0

SEED = 42

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    required_columns = [
        "review_text",
        "aspect",
        "sentiment",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column '{column}' in {path}"
            )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    unknown_labels = (
        set(df["sentiment"])
        - set(LABELS)
    )

    if unknown_labels:

        raise ValueError(
            f"Unknown labels in {path}: "
            f"{unknown_labels}"
        )

    df["label"] = df["sentiment"].map(
        LABEL2ID
    )

    return df


train_df = load_data(TRAIN_PATH)
val_df = load_data(VAL_PATH)
test_df = load_data(TEST_PATH)


# ============================================================
# PRINT DATA INFORMATION
# ============================================================

print("=" * 70)
print("ABSA V4 — DISTILBERT + FOCAL LOSS")
print("=" * 70)

print(
    f"Train examples:      {len(train_df)}"
)

print(
    f"Validation examples: {len(val_df)}"
)

print(
    f"Test examples:       {len(test_df)}"
)

print("\nTraining distribution:")

print(
    train_df["sentiment"].value_counts()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(
        range(len(LABELS))
    ),
    y=train_df["label"].values,
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
)

print("\nClass weights:")

for label, weight in zip(
    LABELS,
    class_weights,
):

    print(
        f"{label:10s}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# ASPECT-AWARE TOKENIZATION
# ============================================================

def tokenize_function(examples):

    texts = []

    for review, aspect in zip(
        examples["review_text"],
        examples["aspect"],
    ):

        text = (
            f"{review} [SEP] {aspect}"
        )

        texts.append(text)

    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[
        [
            "review_text",
            "aspect",
            "label",
        ]
    ],
    preserve_index=False,
)

val_dataset = Dataset.from_pandas(
    val_df[
        [
            "review_text",
            "aspect",
            "label",
        ]
    ],
    preserve_index=False,
)

test_dataset = Dataset.from_pandas(
    test_df[
        [
            "review_text",
            "aspect",
            "label",
        ]
    ],
    preserve_index=False,
)


# ============================================================
# TOKENIZE DATASETS
# ============================================================

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
)


# ============================================================
# FORMAT DATASETS
# ============================================================

columns = [
    "input_ids",
    "attention_mask",
    "label",
]

train_dataset.set_format(
    type="torch",
    columns=columns,
)

val_dataset.set_format(
    type="torch",
    columns=columns,
)

test_dataset.set_format(
    type="torch",
    columns=columns,
)


# ============================================================
# MODEL
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(torch.nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets,
    ):

        log_probs = torch.nn.functional.log_softmax(
            logits,
            dim=-1,
        )

        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        target_probs = probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        focal_factor = (
            1.0 - target_probs
        ) ** self.gamma

        loss = (
            -focal_factor
            * target_log_probs
        )

        if self.alpha is not None:

            alpha = self.alpha.to(
                logits.device
            )

            alpha_t = alpha[targets]

            loss = alpha_t * loss

        return loss.mean()


# ============================================================
# CUSTOM TRAINER
# ============================================================

class FocalLossTrainer(Trainer):

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

        loss_function = FocalLoss(
            alpha=class_weights,
            gamma=FOCAL_GAMMA,
        )

        loss = loss_function(
            logits,
            labels,
        )

        if return_outputs:

            return loss, outputs

        return loss


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    logging_strategy="epoch",

    report_to="none",

    seed=SEED,

    fp16=False,

    dataloader_pin_memory=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = FocalLossTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING V4 TRAINING")
print("=" * 70)

trainer.train()


# ============================================================
# BEST VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BEST VALIDATION RESULTS")
print("=" * 70)

val_metrics = trainer.evaluate(
    eval_dataset=val_dataset
)

for key, value in val_metrics.items():

    if isinstance(value, (float, int)):

        print(
            f"{key}: {float(value):.4f}"
        )


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

test_output = trainer.predict(
    test_dataset
)

test_logits = test_output.predictions

test_predictions = np.argmax(
    test_logits,
    axis=-1,
)

test_true = test_df[
    "label"
].values


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_true,
    test_predictions,
)

test_macro_f1 = f1_score(
    test_true,
    test_predictions,
    average="macro",
    zero_division=0,
)

test_weighted_f1 = f1_score(
    test_true,
    test_predictions,
    average="weighted",
    zero_division=0,
)


print(
    f"Test Accuracy:    "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Macro-F1:    "
    f"{test_macro_f1:.4f}"
)

print(
    f"Test Weighted-F1: "
    f"{test_weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report = classification_report(
    test_true,
    test_predictions,
    labels=list(
        range(len(LABELS))
    ),
    target_names=LABELS,
    zero_division=0,
)

print(report)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "artifacts/absa_v4",
    exist_ok=True,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# SAVE MODEL
# ============================================================

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

output_df = test_df[
    [
        "review_text",
        "aspect",
        "sentiment",
    ]
].copy()

output_df.rename(
    columns={
        "sentiment": "true_sentiment"
    },
    inplace=True,
)

output_df[
    "predicted_sentiment"
] = [
    ID2LABEL[int(pred)]
    for pred in test_predictions
]

output_df.to_csv(
    PREDICTION_PATH,
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "ABSA V4 — DistilBERT + Focal Loss\n"
    )

    f.write(
        "====================================\n"
    )

    f.write(
        f"Model: {MODEL_NAME}\n"
    )

    f.write(
        f"Focal Gamma: {FOCAL_GAMMA}\n"
    )

    f.write(
        f"Learning Rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Epochs: {NUM_EPOCHS}\n"
    )

    f.write(
        f"Max Length: {MAX_LENGTH}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Macro-F1: "
        f"{test_macro_f1:.4f}\n"
    )

    f.write(
        f"Test Weighted-F1: "
        f"{test_weighted_f1:.4f}\n"
    )

    f.write("\nClassification Report:\n")

    f.write(report)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Model saved:       {OUTPUT_DIR}"
)

print(
    f"Predictions saved: "
    f"{PREDICTION_PATH}"
)

print(
    f"Metrics saved:     "
    f"{METRICS_PATH}"
)