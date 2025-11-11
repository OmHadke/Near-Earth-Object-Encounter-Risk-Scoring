#Step 1. Feature Selection
import pandas as pd

data = pd.read_csv("data/cleaned_neo_data.csv")

features = [
    "H", "albedo", "diameter_sigma", "e", "a", "i",
    "absolute_magnitude_h", "estimated_diameter_min_km",
    "estimated_diameter_max_km", "relative_velocity_km_per_s", "miss_distance_km"
]
target = "is_potentially_hazardous"

X = data[features]
y = data[target].astype(int)   # convert True/False → 1/0

# Step 2. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Step 3. Handle Class Imbalance
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

#Step 4. Train Model (Logistic Regression Baseline)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model.fit(X_resampled, y_resampled)

#Step 5. Evaluate Model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Step 6. Generate Risk Scores (0–1)
X_test = X_test.copy()
X_test["true_label"] = y_test.values
X_test["risk_score"] = y_prob

X_test.to_csv("data/neo_risk_scores.csv", index=False)
print("Risk scores saved to data/neo_risk_scores.csv")

#5: “Predicting Hazard Risk for New NEOs”
import joblib

# Save the trained model
joblib.dump(model, "data/neo_risk_model.pkl")

# Also save the column order (so features match later)
joblib.dump(X.columns.tolist(), "data/model_features.pkl")

print("Model saved successfully!")
