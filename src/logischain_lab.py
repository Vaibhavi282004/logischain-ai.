import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data" / "processed"

financial_file = DATA / "financial_risk_dataset.csv"
scenario_file = DATA / "disruption_scenarios.csv"

df = pd.read_csv(financial_file)
scenarios = pd.read_csv(scenario_file)


def simulate_portfolio(scenario):

    result = df.sample(
        n=min(20, len(df)),
        random_state=42
    ).copy()

    delay = scenario["delay_change"]
    freight = scenario["freight_change"]
    inventory = scenario["inventory_change"]

    result["scenario_delay"] = (
        result["carrier_delay"] + delay
    )

    result["scenario_freight"] = (
        result["freight_rate"] * (1 + freight)
    )

    result["scenario_inventory"] = (
        result["inventory_days"] + inventory
    )

    result["scenario_EAD"] = (
        result["scenario_freight"]
        * (1 + result["scenario_inventory"] / 100)
    )

    result["scenario_CCC"] = (
        result["scenario_inventory"]
        + result["DSO"]
        - result["DPO"]
    )

    result["scenario_expected_loss"] = (
        result["PD"]
        * result["LGD"]
        * result["scenario_EAD"]
    )

    result["financial_impact"] = (
        result["scenario_expected_loss"]
        - result["Expected_Loss"]
    )

    return result


print("=" * 60)
print("LOGISCHAIN LAB")
print("=" * 60)

all_results = []

for _, scenario in scenarios.iterrows():

    result = simulate_portfolio(scenario)

    total_loss = result["scenario_expected_loss"].sum()
    total_impact = result["financial_impact"].sum()
    avg_ccc = result["scenario_CCC"].mean()

    all_results.append({
        "scenario_id": scenario["scenario_id"],
        "scenario": scenario["scenario"],
        "portfolio_expected_loss": round(total_loss, 2),
        "financial_impact": round(total_impact, 2),
        "average_CCC": round(avg_ccc, 2)
    })

simulation = pd.DataFrame(all_results)

simulation.to_csv(
    DATA / "logischain_lab_results.csv",
    index=False
)

print("\nSIMULATION RESULTS")
print(simulation.to_string(index=False))

print("\nLogisChain Lab completed.")

print("\nTwo supported game modes:")
print("1. Trade Finance Portfolio Management")
print("2. Supply Chain Finance Pricing")

print("\nSimulation characteristics:")
print("- Portfolio-based exposure analysis")
print("- Disruption-driven financial shocks")
print("- Expected-loss impact")
print("- Working-capital / CCC impact")
print("- Scenario comparison")