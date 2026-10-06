import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILE = BASE_DIR / "data" / "processed" / "logistics_features.csv"


def test_feature_dataset_exists():
    assert FILE.exists()


def test_feature_count():
    df = pd.read_csv(FILE)

    assert len(df.columns) >= 40


def test_rows_exist():
    df = pd.read_csv(FILE)

    assert len(df) > 0