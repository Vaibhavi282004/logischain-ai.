import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILE = BASE_DIR / "data" / "processed" / "financial_risk_dataset.csv"


def test_financial_dataset_exists():
    assert FILE.exists()


def test_financial_columns():
    df = pd.read_csv(FILE)

    required = [
        "PD",
        "LGD",
        "EAD",
        "Expected_Loss",
        "CCC",
        "WCVI",
        "TRFSI",
        "Supply_Chain_Credit_Score"
    ]

    for column in required:
        assert column in df.columns


def test_pd_range():
    df = pd.read_csv(FILE)

    assert df["PD"].between(0, 1).all()


def test_lgd_range():
    df = pd.read_csv(FILE)

    assert df["LGD"].between(0, 1).all()


def test_expected_loss_nonnegative():
    df = pd.read_csv(FILE)

    assert (df["Expected_Loss"] >= 0).all()