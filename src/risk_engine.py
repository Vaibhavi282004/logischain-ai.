import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("models/shipment_risk_model.pkl")

model = joblib.load(MODEL_PATH)


def calculate_risk(shipment_data):
    """
    Predict shipment disruption risk and estimate financial impact.
    """

    features = [
        "distance_km",
        "freight_rate",
        "port_congestion",
        "carrier_delay",
        "inventory_days",
        "supplier_risk",
        "route_risk",
        "demand_change"
    ]

    input_data = pd.DataFrame([shipment_data])[features]

    # Probability of disruption
    probability = model.predict_proba(input_data)[0][1]

    # Risk category
    if probability < 0.35:
        risk_level = "LOW"
    elif probability < 0.65:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # Simple financial exposure estimate
    exposure = (
        shipment_data["freight_rate"]
        * (1 + shipment_data["port_congestion"] / 10)
    )

    expected_loss = exposure * probability

    return {
        "disruption_probability": probability,
        "risk_level": risk_level,
        "financial_exposure": exposure,
        "expected_loss": expected_loss
    }


if __name__ == "__main__":

    example_shipment = {
        "distance_km": 2500,
        "freight_rate": 2500,
        "port_congestion": 7.5,
        "carrier_delay": 8,
        "inventory_days": 30,
        "supplier_risk": 0.65,
        "route_risk": 0.55,
        "demand_change": -0.15
    }

    result = calculate_risk(example_shipment)

    print("\n===== LogisChain AI Risk Engine =====")
    print(f"Disruption Probability: {result['disruption_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Financial Exposure: ${result['financial_exposure']:,.2f}")
    print(f"Expected Loss: ${result['expected_loss']:,.2f}")