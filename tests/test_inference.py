from pathlib import Path
import subprocess, sys

def test_training_pipeline():
    subprocess.run([sys.executable, "scripts/prepare_demo_data.py"], check=True)
    subprocess.run([
        sys.executable, "-m", "src.train",
        "--model", "tfidf",
        "--data", "data/processed/demo_reviews.csv"
    ], check=True)
    assert Path("artifacts/complaint_model.joblib").exists()
