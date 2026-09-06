
from pathlib import Path
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "artifacts" / "absa_v4" / "model"
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "semeval" / "test.csv"
OUTPUT_FILE = PROJECT_ROOT / "artifacts" / "absa_v4" / "final_inference.csv"


# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------
LABELS = {
    0: "positive",
    1: "negative",
    2: "neutral",
    3: "conflict",
}


# ---------------------------------------------------------
# Complaint detection
# Transparent rule-based baseline
# ---------------------------------------------------------
COMPLAINT_PATTERNS = {
    "battery_power": [
        "battery",
        "battery life",
        "battery went",
        "battery died",
        "doesn't hold",
        "does not hold",
        "drain",
        "drained",
        "juice",
    ],
    "display": [
        "screen",
        "display",
        "resolution",
        "brightness",
        "pixel",
        "flicker",
    ],
    "input_device": [
        "keyboard",
        "key pad",
        "keypad",
        "trackpad",
        "touchpad",
        "mouse",
        "touch screen",
    ],
    "shipping": [
        "shipping",
        "delivery",
        "delivered",
        "shipment",
    ],
    "service_warranty": [
        "warranty",
        "warrenty",
        "refund",
        "exchange",
        "customer service",
        "service",
    ],
    "performance_reliability": [
        "not working",
        "does not work",
        "doesn't work",
        "failed",
        "fails",
        "failure",
        "problem",
        "problems",
        "issue",
        "issues",
        "defect",
        "defective",
        "broken",
        "stopped working",
        "too slow",
        "overheat",
        "overheats",
        "heated up",
        "crash",
        "crashes",
        "freeze",
        "freezes",
    ],
    "price_cost": [
        "expensive",
        "overpriced",
        "costly",
        "price",
        "charged",
        "cost",
    ],
}


GENERAL_COMPLAINT_PATTERNS = [
    "not working",
    "does not work",
    "doesn't work",
    "failed",
    "fails",
    "failure",
    "problem",
    "problems",
    "issue",
    "issues",
    "defect",
    "defective",
    "broken",
    "stopped working",
    "refund",
    "exchange",
    "complaint",
    "complain",
    "unable",
    "cannot",
    "can't",
    "disappointed",
    "disappointing",
    "terrible",
    "worst",
    "poor",
]


NEGATIVE_INDICATORS = [
    "not",
    "no",
    "never",
    "failed",
    "failure",
    "problem",
    "problems",
    "issue",
    "issues",
    "broken",
    "stopped",
    "slow",
    "overheat",
    "heated",
    "expensive",
    "poor",
    "terrible",
    "worst",
    "disappointed",
    "disappointing",
    "sucked",
    "bad",
]


def normalize_text(text):
    return str(text).lower().strip()


def detect_complaint(review_text, aspect, predicted_sentiment):
    """
    Rule-based complaint baseline.

    Important:
    SemEval does not contain a true complaint-intent label.
    Therefore this is NOT evaluated as supervised complaint accuracy.

    We primarily inspect the aspect-specific context and use
    predicted sentiment as an additional signal.
    """

    text = normalize_text(review_text)
    aspect_text = normalize_text(aspect)

    # Find aspect occurrence in the review.
    aspect_position = text.find(aspect_text)

    if aspect_position >= 0:
        start = max(0, aspect_position - 120)
        end = min(len(text), aspect_position + len(aspect_text) + 180)
        context = text[start:end]
    else:
        context = text

    matched_reasons = []

    for reason, patterns in COMPLAINT_PATTERNS.items():
        for pattern in patterns:
            if pattern in context:
                matched_reasons.append(reason)
                break

    # Explicit complaint language is strong evidence.
    explicit_complaint = any(
        pattern in context for pattern in GENERAL_COMPLAINT_PATTERNS
    )

    # Negative sentiment is useful evidence, but negative sentiment
    # alone should NOT automatically become a complaint.
    negative_prediction = predicted_sentiment == "negative"

    # Conflict is potentially complaint-relevant when explicit
    # complaint/problem language exists.
    conflict_prediction = predicted_sentiment == "conflict"

    # Complaint decision.
    if explicit_complaint:
        complaint = True
    elif matched_reasons and (negative_prediction or conflict_prediction):
        complaint = True
    elif matched_reasons and any(
        indicator in context for indicator in NEGATIVE_INDICATORS
    ):
        complaint = True
    else:
        complaint = False

    if complaint and matched_reasons:
        # Preserve first matched reason.
        reason = matched_reasons[0]
    elif complaint:
        reason = "other"
    else:
        reason = "none"

    return complaint, reason


# ---------------------------------------------------------
# V4 ABSA model
# ---------------------------------------------------------
class V4ABSAInference:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"V4 model not found at: {model_dir}\n"
                "Make sure artifacts\\absa_v4\\model exists."
            )

        print(f"Loading V4 model from: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir
        )

        self.model.eval()

        self.device = torch.device("cpu")
        self.model.to(self.device)

        print("V4 model loaded successfully.")

    def predict(self, review_text, aspect):
        text = f"{review_text} [SEP] {aspect}"

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_id].item())

        sentiment = LABELS.get(
            predicted_id,
            self.model.config.id2label.get(
                predicted_id,
                str(predicted_id),
            ),
        )

        return sentiment, confidence


# ---------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------
def main():
    print("=" * 70)
    print("CUSTOMER REVIEW INTELLIGENCE - FINAL INFERENCE")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(f"\nInput dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "review_text",
        "aspect",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"Total examples: {len(df)}")

    predictor = V4ABSAInference(MODEL_DIR)

    results = []

    print("\nRunning V4 ABSA inference...")

    for index, row in df.iterrows():
        review_text = str(row["review_text"])
        aspect = str(row["aspect"])

        sentiment, confidence = predictor.predict(
            review_text,
            aspect,
        )

        complaint, reason = detect_complaint(
            review_text,
            aspect,
            sentiment,
        )

        results.append(
            {
                "review_text": review_text,
                "aspect": aspect,
                "predicted_sentiment": sentiment,
                "sentiment_confidence": round(confidence, 4),
                "complaint": complaint,
                "complaint_reason": reason,
            }
        )

        if (index + 1) % 50 == 0:
            print(
                f"Processed {index + 1}/{len(df)} examples..."
            )

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(f"\nOutput file:")
    print(OUTPUT_FILE)

    print("\nSentiment distribution:")
    print(
        result_df["predicted_sentiment"]
        .value_counts()
        .to_string()
    )

    print("\nComplaint distribution:")
    print(
        result_df["complaint"]
        .value_counts()
        .to_string()
    )

    print("\nComplaint reasons:")
    print(
        result_df.loc[
            result_df["complaint"] == True,
            "complaint_reason",
        ]
        .value_counts()
        .to_string()
    )

    print("\nSample predictions:")
    print(
        result_df[
            [
                "aspect",
                "predicted_sentiment",
                "sentiment_confidence",
                "complaint",
                "complaint_reason",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nFinal output saved successfully.")


if __name__ == "__main__":
    main()