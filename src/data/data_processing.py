import pandas as pd
import numpy as np
from pathlib import Path

# Reproducible results
np.random.seed(42)

# Number of shipments
n = 2000

data = pd.DataFrame({
    "shipment_id": range(1, n + 1),

    "distance_km": np.random.randint(100, 5000, n),

    "freight_rate": np.round(
        np.random.uniform(500, 5000, n), 2
    ),

    "port_congestion": np.round(
        np.random.uniform(0, 10, n), 2
    ),

    "carrier_delay": np.random.randint(0, 15, n),

    "inventory_days": np.random.randint(5, 90, n),

    "supplier_risk": np.round(
        np.random.uniform(0, 1, n), 2
    ),

    "route_risk": np.round(
        np.random.uniform(0, 1, n), 2
    ),

    "demand_change": np.round(
        np.random.uniform(-0.5, 0.5, n), 2
    )
})

# Create a disruption probability
risk_score = (
    0.20 * (data["port_congestion"] / 10)
    + 0.20 * (data["carrier_delay"] / 15)
    + 0.20 * data["supplier_risk"]
    + 0.20 * data["route_risk"]
    + 0.10 * abs(data["demand_change"])
    + 0.10 * (data["inventory_days"] / 90)
)

# Add small randomness
risk_score += np.random.normal(0, 0.08, n)

# Convert to binary target
data["shipment_disrupted"] = (risk_score > 0.50).astype(int)

# Save dataset
output_path = Path("data/raw/logistics_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

data.to_csv(output_path, index=False)

print("Dataset created successfully!")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 rows:")
print(data.head())

print("\nDisruption distribution:")
print(data["shipment_disrupted"].value_counts())