import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "logistics_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "financial_risk_dataset.csv"


def calculate_financial_risk(df):

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Probability of Default (PD)
    # ---------------------------------------------------------
    #
    # This is a supply-chain risk proxy, not a bank-calibrated PD.
    #

    risk_score = (
        0.25 * df["supplier_risk"]
        + 0.20 * df["route_risk"]
        + 0.20 * df["port_congestion"]
        + 0.15 * df["carrier_delay"]
        + 0.10 * df["demand_change"].abs()
        + 0.10 * df["inventory_days_normalized"]
    )

    df["PD"] = np.clip(risk_score, 0.001, 0.95)

    # ---------------------------------------------------------
    # 2. Loss Given Default (LGD)
    # ---------------------------------------------------------

    df["LGD"] = np.clip(
        0.35
        + 0.20 * df["supplier_risk"]
        + 0.15 * df["route_risk"]
        + 0.10 * df["port_congestion"],
        0.20,
        0.90
    )

    # ---------------------------------------------------------
    # 3. Exposure at Default (EAD)
    # ---------------------------------------------------------

    df["EAD"] = (
        df["freight_rate"]
        * (1 + df["distance_km"] / 10000)
        * (1 + df["inventory_days"] / 100)
    )

    # ---------------------------------------------------------
    # 4. Expected Loss
    # ---------------------------------------------------------

    df["Expected_Loss"] = (
        df["PD"]
        * df["LGD"]
        * df["EAD"]
    )

    # ---------------------------------------------------------
    # 5. Risk-adjusted exposure
    # ---------------------------------------------------------

    df["Risk_Adjusted_Exposure"] = (
        df["EAD"]
        + df["Expected_Loss"]
    )

    # ---------------------------------------------------------
    # 6. Cash Conversion Cycle
    # ---------------------------------------------------------
    #
    # Simplified logistics-oriented CCC proxy:
    #
    # DIO = Days Inventory Outstanding
    # DSO = Days Sales Outstanding
    # DPO = Days Payable Outstanding
    #

    df["DIO"] = (
        df["inventory_days"]
        * (1 + df["demand_change"].abs())
    )

    df["DSO"] = (
        30
        + 10 * df["carrier_delay"]
        + 10 * df["port_congestion"]
    )

    df["DPO"] = (
        30
        - 10 * df["supplier_risk"]
        - 5 * df["carrier_delay"]
    )

    df["CCC"] = (
        df["DIO"]
        + df["DSO"]
        - df["DPO"]
    )

    # ---------------------------------------------------------
    # 7. Working Capital Vulnerability Index (WCVI)
    # ---------------------------------------------------------

    df["WCVI"] = (
        0.40 * df["CCC"] / 100
        + 0.20 * df["carrier_delay"]
        + 0.20 * df["port_congestion"]
        + 0.20 * df["demand_change"].abs()
    )

    # ---------------------------------------------------------
    # 8. Supply Chain Financial Risk Index
    # ---------------------------------------------------------

    df["SC_Financial_Risk_Index"] = (
        0.35 * df["PD"]
        + 0.25 * df["LGD"]
        + 0.20 * df["WCVI"]
        + 0.20 * df["supplier_risk"]
    )

    # ---------------------------------------------------------
    # 9. Trade Risk Financing Stress Index
    # ---------------------------------------------------------

    df["TRFSI"] = (
        0.30 * df["Expected_Loss"]
        + 0.25 * df["WCVI"]
        + 0.20 * df["port_congestion"]
        + 0.15 * df["carrier_delay"]
        + 0.10 * df["route_risk"]
    )

    # ---------------------------------------------------------
    # 10. Supply Chain Credit Score
    # ---------------------------------------------------------

    df["Supply_Chain_Credit_Score"] = np.clip(
        100
        - (
            45 * df["PD"]
            + 20 * df["LGD"]
            + 15 * df["WCVI"]
            + 20 * df["supplier_risk"]
        ),
        0,
        100
    )

    # ---------------------------------------------------------
    # 11. Financial risk category
    # ---------------------------------------------------------

    def classify_risk(value):

        if value < 0.30:
            return "LOW"

        elif value < 0.60:
            return "MEDIUM"

        else:
            return "HIGH"

    df["Financial_Risk_Level"] = (
        df["SC_Financial_Risk_Index"]
        .apply(classify_risk)
    )

    return df


def main():

    print("=" * 60)
    print("LOGISCHAIN AI - FINANCIAL RISK ENGINE")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded {len(df)} records.")

    result = calculate_financial_risk(df)

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFinancial risk calculations completed.")

    print("\nKey metrics:")
    print(
        result[
            [
                "PD",
                "LGD",
                "EAD",
                "Expected_Loss",
                "CCC",
                "WCVI",
                "TRFSI",
                "Supply_Chain_Credit_Score"
            ]
        ].describe()
    )

    print("\nFinancial risk distribution:")

    print(
        result["Financial_Risk_Level"]
        .value_counts()
    )

    print("\nSample records:")

    print(
        result[
            [
                "shipment_id",
                "PD",
                "LGD",
                "EAD",
                "Expected_Loss",
                "CCC",
                "WCVI",
                "Supply_Chain_Credit_Score",
                "Financial_Risk_Level"
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()