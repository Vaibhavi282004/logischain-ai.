import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "logistics_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "logistics_features.csv"


def create_features(df):

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Basic ratios
    # ---------------------------------------------------------

    df["freight_per_km"] = df["freight_rate"] / (df["distance_km"] + 1)
    df["delay_per_1000km"] = df["carrier_delay"] / (df["distance_km"] / 1000 + 1)
    df["congestion_delay_ratio"] = df["port_congestion"] * df["carrier_delay"]
    df["supplier_route_risk"] = df["supplier_risk"] * df["route_risk"]

    # ---------------------------------------------------------
    # 2. Operational risk
    # ---------------------------------------------------------

    df["operational_risk"] = (
        0.30 * df["port_congestion"]
        + 0.25 * df["carrier_delay"]
        + 0.25 * df["supplier_risk"]
        + 0.20 * df["route_risk"]
    )

    df["network_risk"] = (
        df["port_congestion"]
        + df["route_risk"]
        + df["supplier_risk"]
    ) / 3

    df["delay_risk"] = (
        df["carrier_delay"] * 0.6
        + df["port_congestion"] * 0.4
    )

    df["inventory_risk"] = (
        df["inventory_days"] * df["demand_change"].abs()
    )

    df["demand_risk"] = df["demand_change"].abs() * df["supplier_risk"]

    # ---------------------------------------------------------
    # 3. Financial exposure indicators
    # ---------------------------------------------------------

    df["base_exposure"] = df["freight_rate"] * (
        1 + df["distance_km"] / 10000
    )

    df["congestion_exposure"] = (
        df["freight_rate"] * df["port_congestion"]
    )

    df["delay_exposure"] = (
        df["freight_rate"] * df["carrier_delay"]
    )

    df["inventory_exposure"] = (
        df["freight_rate"] * df["inventory_days"]
    )

    df["total_operational_exposure"] = (
        df["base_exposure"]
        + df["congestion_exposure"]
        + df["delay_exposure"]
    )

    # ---------------------------------------------------------
    # 4. Supply-chain stress indicators
    # ---------------------------------------------------------

    df["supply_chain_stress"] = (
        df["port_congestion"]
        + df["carrier_delay"]
        + df["supplier_risk"]
        + df["route_risk"]
        + df["demand_change"].abs()
    )

    df["high_risk_interaction"] = (
        df["supplier_risk"]
        * df["route_risk"]
        * (1 + df["port_congestion"])
    )

    df["disruption_pressure"] = (
        df["carrier_delay"]
        + df["port_congestion"]
        + df["demand_change"].abs()
    )

    df["supplier_disruption_score"] = (
        df["supplier_risk"] * df["carrier_delay"]
    )

    df["route_disruption_score"] = (
        df["route_risk"] * df["port_congestion"]
    )

    # ---------------------------------------------------------
    # 5. Inventory / working-capital indicators
    # ---------------------------------------------------------

    df["inventory_buffer"] = df["inventory_days"] / (
        1 + df["carrier_delay"]
    )

    df["inventory_pressure"] = (
        df["inventory_days"]
        * (1 + df["demand_change"].abs())
    )

    df["working_capital_pressure"] = (
        df["inventory_days"]
        + df["carrier_delay"]
        + df["port_congestion"]
    )

    df["cash_cycle_pressure"] = (
        df["inventory_days"]
        * (1 + df["supplier_risk"])
    )

    # ---------------------------------------------------------
    # 6. Route efficiency
    # ---------------------------------------------------------

    df["route_efficiency"] = (
        df["distance_km"]
        / (1 + df["carrier_delay"])
    )

    df["adjusted_distance"] = (
        df["distance_km"]
        * (1 + df["route_risk"])
    )

    df["risk_adjusted_freight"] = (
        df["freight_rate"]
        * (1 + df["route_risk"])
        * (1 + df["port_congestion"])
    )

    df["transport_cost_pressure"] = (
        df["freight_rate"]
        * (1 + df["carrier_delay"])
    )

    # ---------------------------------------------------------
    # 7. Composite indicators
    # ---------------------------------------------------------

    df["shipment_risk_index"] = (
        0.20 * df["port_congestion"]
        + 0.20 * df["carrier_delay"]
        + 0.20 * df["supplier_risk"]
        + 0.20 * df["route_risk"]
        + 0.10 * df["demand_change"].abs()
        + 0.10 * (df["inventory_days"] / 100)
    )

    df["logistics_resilience"] = 1 / (
        1 + df["supply_chain_stress"]
    )

    df["financial_stress_index"] = (
        df["risk_adjusted_freight"]
        + df["working_capital_pressure"]
    )

    df["disruption_risk_index"] = (
        df["shipment_risk_index"]
        + df["disruption_pressure"] / 10
    )

    # ---------------------------------------------------------
    # 8. Threshold flags
    # ---------------------------------------------------------

    df["high_congestion_flag"] = (
        df["port_congestion"] > 0.7
    ).astype(int)

    df["high_delay_flag"] = (
        df["carrier_delay"] > 0.7
    ).astype(int)

    df["high_supplier_risk_flag"] = (
        df["supplier_risk"] > 0.7
    ).astype(int)

    df["high_route_risk_flag"] = (
        df["route_risk"] > 0.7
    ).astype(int)

    df["high_demand_change_flag"] = (
        df["demand_change"].abs() > 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # 9. Interaction features
    # ---------------------------------------------------------

    df["distance_freight_interaction"] = (
        df["distance_km"] * df["freight_rate"]
    )

    df["distance_risk_interaction"] = (
        df["distance_km"] * df["route_risk"]
    )

    df["freight_delay_interaction"] = (
        df["freight_rate"] * df["carrier_delay"]
    )

    df["freight_congestion_interaction"] = (
        df["freight_rate"] * df["port_congestion"]
    )

    df["inventory_supplier_interaction"] = (
        df["inventory_days"] * df["supplier_risk"]
    )

    df["inventory_route_interaction"] = (
        df["inventory_days"] * df["route_risk"]
    )

    df["demand_inventory_interaction"] = (
        df["demand_change"] * df["inventory_days"]
    )

    # ---------------------------------------------------------
    # 10. Normalized indicators
    # ---------------------------------------------------------

    for column in [
        "distance_km",
        "freight_rate",
        "port_congestion",
        "carrier_delay",
        "inventory_days",
        "supplier_risk",
        "route_risk",
        "demand_change"
    ]:
        min_value = df[column].min()
        max_value = df[column].max()

        df[column + "_normalized"] = (
            (df[column] - min_value)
            / (max_value - min_value + 1e-9)
        )

    # ---------------------------------------------------------
    # Clean data
    # ---------------------------------------------------------

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def main():

    print("Loading logistics dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    df = create_features(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature engineering completed.")
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()