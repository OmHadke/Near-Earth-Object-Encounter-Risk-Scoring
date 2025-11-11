from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and feature names
model = joblib.load("data/neo_risk_model.pkl")
features = joblib.load("data/model_features.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from the form
        user_input = {f: float(request.form[f]) for f in features}
        df = pd.DataFrame([user_input])
        
        # Predict
        prob = model.predict_proba(df)[:, 1][0]
        pred = model.predict(df)[0]
        
        result = "🚨 Hazardous" if pred == 1 else "✅ Safe"
        risk = f"{prob:.3f}"
        color = "red" if pred == 1 else "green"
        
        return render_template("index.html", result=result, risk=risk, color=color)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
