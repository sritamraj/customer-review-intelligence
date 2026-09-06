from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "semeval_source"
    / "semeval_data"
    / "Laptop_Train_v2.xml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semeval_laptop_absa.csv"
)


REQUIRED_COLUMNS = [
    "review_text",
    "aspect",
    "sentiment",
    "from",
    "to",
]


def parse_semeval(xml_path: Path) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()

    rows = []

    for sentence in root.findall(".//sentence"):
        text = sentence.findtext("text", default="")

        for aspect in sentence.findall("./aspectTerms/aspectTerm"):
            rows.append(
                {
                    "review_text": text,
                    "aspect": aspect.attrib.get("term", ""),
                    "sentiment": aspect.attrib.get("polarity", ""),
                    "from": int(aspect.attrib.get("from", -1)),
                    "to": int(aspect.attrib.get("to", -1)),
                }
            )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def validate_dataset(df: pd.DataFrame) -> None:
    missing = df[REQUIRED_COLUMNS].isna().sum()

    if missing.any():
        raise ValueError(f"Missing values found:\n{missing}")

    valid_sentiments = {
        "positive",
        "negative",
        "neutral",
        "conflict",
    }

    unexpected = set(df["sentiment"]) - valid_sentiments

    if unexpected:
        raise ValueError(
            f"Unexpected sentiment labels: {unexpected}"
        )

    offset_errors = 0

    for _, row in df.iterrows():
        extracted = row["review_text"][
            int(row["from"]):int(row["to"])
        ]

        if extracted != row["aspect"]:
            offset_errors += 1

    if offset_errors > 0:
        raise ValueError(
            f"Found {offset_errors} invalid aspect offsets."
        )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = parse_semeval(INPUT_FILE)

    rows_before = len(df)

    validate_dataset(df)

    df = df.drop_duplicates().reset_index(drop=True)

    rows_after = len(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows before deduplication: {rows_before}")
    print(f"Duplicate rows removed: {rows_before - rows_after}")
    print(f"Rows after deduplication: {rows_after}")

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nUnique aspects:")
    print(df["aspect"].nunique())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()