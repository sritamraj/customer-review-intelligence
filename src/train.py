import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from .data import load_reviews
from .models import TfidfComplaintModel
from .utils import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tfidf"], default="tfidf")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/complaint_model.joblib")
    args = parser.parse_args()

    set_seed(42)
    df = load_reviews(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        df["review"], df["complaint"],
        test_size=0.2,
        random_state=42,
        stratify=df["complaint"]
    )

    model = TfidfComplaintModel.build()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    print(classification_report(y_test, pred, digits=4))
    print("macro_f1:", round(f1_score(y_test, pred, average="macro"), 4))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)

    pd.DataFrame({"review": x_test, "label": y_test, "prediction": pred}).to_csv(
        "artifacts/predictions.csv", index=False
    )

if __name__ == "__main__":
    main()
