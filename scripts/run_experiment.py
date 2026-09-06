from pathlib import Path
import subprocess
import sys

data = "data/processed/demo_reviews.csv"
if not Path(data).exists():
    subprocess.run([sys.executable, "scripts/prepare_demo_data.py"], check=True)

subprocess.run([
    sys.executable, "-m", "src.train",
    "--model", "tfidf",
    "--data", data
], check=True)
