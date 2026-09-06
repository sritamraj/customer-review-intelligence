import re
import pandas as pd


# ============================================================
# COMPLAINT DETECTION BASELINE
# ============================================================

COMPLAINT_PATTERNS = [
    r"\bnot working\b",
    r"\bdoes not work\b",
    r"\bdoesn't work\b",
    r"\bfailed\b",
    r"\bfails\b",
    r"\bfailure\b",
    r"\bproblem\b",
    r"\bproblems\b",
    r"\bissue\b",
    r"\bissues\b",
    r"\bdefect\b",
    r"\bdefective\b",
    r"\bbroken\b",
    r"\bstopped working\b",
    r"\btoo slow\b",
    r"\bslow\b",
    r"\boverheat\b",
    r"\boverheats\b",
    r"\bheated up\b",
    r"\brefund\b",
    r"\bexchange\b",
    r"\bwarranty\b",
    r"\bwarrenty\b",
    r"\bexpensive\b",
    r"\bcharged\b",
    r"\bcharge\b",
    r"\bpoor\b",
    r"\bterrible\b",
    r"\bworst\b",
    r"\bdisappointed\b",
    r"\bdisappointing\b",
    r"\bcomplaint\b",
    r"\bcomplain\b",
    r"\bdoes not\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\bunable\b",
]


def detect_complaint(review_text, sentiment=None):
    """
    Rule-based complaint detection baseline.

    A review is considered complaint-like when:
    1. It contains a complaint-related pattern, OR
    2. The sentiment is negative and the review contains
       a problem/failure-related expression.

    Returns:
        complaint_label: YES / NO
        matched_patterns: list of matched patterns
    """

    text = str(review_text).lower()

    matched_patterns = []

    for pattern in COMPLAINT_PATTERNS:

        if re.search(pattern, text):

            matched_patterns.append(pattern)

    if matched_patterns:

        return "YES", matched_patterns

    return "NO", []


def classify_complaint_reason(review_text):
    """
    Assign a broad complaint reason using interpretable rules.
    """

    text = str(review_text).lower()

    if any(
        word in text
        for word in [
            "battery",
            "battery life",
            "charge",
            "charger",
        ]
    ):
        return "battery_power"

    if any(
        word in text
        for word in [
            "screen",
            "display",
            "resolution",
        ]
    ):
        return "display"

    if any(
        word in text
        for word in [
            "keyboard",
            "key pad",
            "trackpad",
            "mouse",
        ]
    ):
        return "input_device"

    if any(
        word in text
        for word in [
            "shipping",
            "delivery",
        ]
    ):
        return "shipping"

    if any(
        word in text
        for word in [
            "warranty",
            "warrenty",
            "refund",
            "exchange",
            "service center",
            "customer service",
        ]
    ):
        return "service_warranty"

    if any(
        word in text
        for word in [
            "slow",
            "performance",
            "freeze",
            "froze",
            "crash",
            "reboot",
        ]
    ):
        return "performance_reliability"

    if any(
        word in text
        for word in [
            "price",
            "expensive",
            "cost",
            "charged",
        ]
    ):
        return "price_cost"

    return "other"


# ============================================================
# MAIN
# ============================================================

def main():

    input_path = (
        "artifacts/absa_v4/test_predictions.csv"
    )

    output_path = (
        "artifacts/absa_v4/complaint_predictions.csv"
    )

    df = pd.read_csv(input_path)

    required_columns = [
        "review_text",
        "aspect",
        "true_sentiment",
        "predicted_sentiment",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    complaint_results = []

    for _, row in df.iterrows():

        complaint_label, matched = (
            detect_complaint(
                row["review_text"],
                row["predicted_sentiment"],
            )
        )

        reason = classify_complaint_reason(
            row["review_text"]
        )

        complaint_results.append(
            {
                "complaint": complaint_label,
                "complaint_reason": reason,
                "matched_patterns": "; ".join(
                    matched
                ),
            }
        )

    complaint_df = pd.DataFrame(
        complaint_results
    )

    output_df = pd.concat(
        [
            df.reset_index(drop=True),
            complaint_df,
        ],
        axis=1,
    )

    output_df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("COMPLAINT DETECTION BASELINE")
    print("=" * 70)

    print(
        f"Total examples: {len(output_df)}"
    )

    print("\nComplaint distribution:")

    print(
        output_df["complaint"].value_counts()
    )

    print("\nComplaint reasons:")

    print(
        output_df[
            output_df["complaint"] == "YES"
        ]["complaint_reason"].value_counts()
    )

    print("\nSample complaint detections:")

    samples = output_df[
        output_df["complaint"] == "YES"
    ].head(10)

    for _, row in samples.iterrows():

        print("\n" + "-" * 70)

        print(
            f"Aspect: {row['aspect']}"
        )

        print(
            f"Sentiment: "
            f"{row['predicted_sentiment']}"
        )

        print(
            f"Complaint: "
            f"{row['complaint']}"
        )

        print(
            f"Reason: "
            f"{row['complaint_reason']}"
        )

        print(
            f"Review: "
            f"{row['review_text']}"
        )

    print("\n" + "=" * 70)

    print(
        f"Saved: {output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()