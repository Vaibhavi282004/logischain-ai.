import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED = BASE_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# =========================================================
# SUPPLY-CHAIN NETWORK
# =========================================================

np.random.seed(42)

countries = [
    "India", "China", "USA", "Germany", "Japan",
    "Singapore", "Netherlands", "UAE", "UK", "France",
    "South Korea", "Vietnam", "Australia", "Brazil",
    "Canada", "Italy", "Spain", "Malaysia", "Thailand", "Indonesia"
]

ports = [
    "Mumbai", "Shanghai", "Los Angeles", "Hamburg",
    "Tokyo", "Singapore", "Rotterdam", "Dubai",
    "London", "Singapore-West", "Busan", "Ho Chi Minh",
    "Sydney", "Santos", "Vancouver", "Genoa",
    "Valencia", "Port Klang", "Laem Chabang", "Jakarta"
]

nodes = []

for i, country in enumerate(countries):
    nodes.append({
        "node_id": f"C{i+1:03d}",
        "node_type": "COUNTRY",
        "name": country,
        "risk_score": round(np.random.uniform(0.10, 0.80), 3)
    })

for i, port in enumerate(ports):
    nodes.append({
        "node_id": f"P{i+1:03d}",
        "node_type": "PORT",
        "name": port,
        "risk_score": round(np.random.uniform(0.10, 0.90), 3)
    })

nodes_df = pd.DataFrame(nodes)

nodes_df.to_csv(
    PROCESSED / "network_nodes.csv",
    index=False
)

# Create network links

links = []

for i in range(60):
    source = np.random.choice(nodes_df["node_id"])
    target = np.random.choice(nodes_df["node_id"])

    if source == target:
        continue

    links.append({
        "link_id": f"L{i+1:03d}",
        "source": source,
        "target": target,
        "link_type": np.random.choice([
            "TRADE",
            "SHIPPING",
            "SUPPLY",
            "FINANCE"
        ]),
        "capacity": round(np.random.uniform(50, 1000), 2),
        "risk": round(np.random.uniform(0.05, 0.90), 3)
    })

links_df = pd.DataFrame(links)

links_df.to_csv(
    PROCESSED / "network_links.csv",
    index=False
)

print("=" * 60)
print("LOGISCHAIN AI - SUPPLY CHAIN NETWORK")
print("=" * 60)

print(f"Nodes created : {len(nodes_df)}")
print(f"Links created : {len(links_df)}")

# =========================================================
# DISRUPTION SCENARIOS
# =========================================================

scenarios = [
    {
        "scenario_id": "S01",
        "scenario": "Port Congestion",
        "description": "Major port congestion increases shipment delays.",
        "delay_change": 0.35,
        "freight_change": 0.25,
        "inventory_change": 15
    },
    {
        "scenario_id": "S02",
        "scenario": "Supplier Failure",
        "description": "Critical supplier experiences operational failure.",
        "delay_change": 0.40,
        "freight_change": 0.20,
        "inventory_change": 20
    },
    {
        "scenario_id": "S03",
        "scenario": "Geopolitical Disruption",
        "description": "Trade route disruption increases route risk.",
        "delay_change": 0.45,
        "freight_change": 0.35,
        "inventory_change": 25
    },
    {
        "scenario_id": "S04",
        "scenario": "Demand Shock",
        "description": "Sudden demand decline creates inventory pressure.",
        "delay_change": 0.10,
        "freight_change": 0.15,
        "inventory_change": 30
    },
    {
        "scenario_id": "S05",
        "scenario": "Carrier Disruption",
        "description": "Carrier reliability deteriorates.",
        "delay_change": 0.50,
        "freight_change": 0.30,
        "inventory_change": 20
    },
    {
        "scenario_id": "S06",
        "scenario": "Fuel Cost Shock",
        "description": "Transportation costs increase significantly.",
        "delay_change": 0.10,
        "freight_change": 0.45,
        "inventory_change": 10
    },
    {
        "scenario_id": "S07",
        "scenario": "Working Capital Stress",
        "description": "Longer inventory cycles increase financing pressure.",
        "delay_change": 0.25,
        "freight_change": 0.20,
        "inventory_change": 35
    },
    {
        "scenario_id": "S08",
        "scenario": "Multi-node Disruption",
        "description": "Several supply-chain nodes experience simultaneous disruption.",
        "delay_change": 0.55,
        "freight_change": 0.40,
        "inventory_change": 40
    }
]

scenarios_df = pd.DataFrame(scenarios)

scenarios_df.to_csv(
    PROCESSED / "disruption_scenarios.csv",
    index=False
)

print("\nDisruption scenarios created:", len(scenarios_df))

print("\nScenarios:")
print(
    scenarios_df[
        ["scenario_id", "scenario", "description"]
    ].to_string(index=False)
)

print("\nFiles created:")
print(PROCESSED / "network_nodes.csv")
print(PROCESSED / "network_links.csv")
print(PROCESSED / "disruption_scenarios.csv")