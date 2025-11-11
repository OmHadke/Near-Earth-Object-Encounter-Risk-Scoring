#This script loads your saved model and takes new asteroid parameters as input.
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("data/neo_risk_model.pkl")
features = joblib.load("data/model_features.pkl")

def predict_new_neo(params):
    """Predict hazard risk score for a new NEO."""
    df = pd.DataFrame([params])[features]
    risk_score = model.predict_proba(df)[:, 1][0]
    prediction = model.predict(df)[0]
    return prediction, risk_score

# Example usage
new_neo = {
    "H": 21.3,
    "albedo": 0.15,
    "diameter_sigma": 0.2,
    "e": 0.47,
    "a": 1.18,
    "i": 5.2,
    "absolute_magnitude_h": 21.3,
    "estimated_diameter_min_km": 0.12,
    "estimated_diameter_max_km": 0.27,
    "relative_velocity_km_per_s": 12.8,
    "miss_distance_km": 1.9e7
}

pred, score = predict_new_neo(new_neo)
print(f"Predicted class: {'Hazardous' if pred==1 else 'Safe'}")
print(f"Risk score: {score:.3f}")
