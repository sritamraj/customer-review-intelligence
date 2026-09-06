from pathlib import Path
from src.data import load_reviews

def test_demo_data_schema():
    path = Path("data/processed/demo_reviews.csv")
    if not path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, "scripts/prepare_demo_data.py"], check=True)
    df = load_reviews(path)
    assert {"review", "complaint", "complaint_category"} <= set(df.columns)
    assert len(df) > 0
