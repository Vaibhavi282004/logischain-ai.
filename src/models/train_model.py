import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Load dataset
data_path = Path("data/raw/logistics_data.csv")
data = pd.read_csv(data_path)

# Features
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

X = data[features]
y = data["shipment_disrupted"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n===== LogisChain AI Model =====")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model
model_path = Path("models/shipment_risk_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")