from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import math
from typing import Any

app = Flask(__name__)

# Load model and features (be tolerant if files are missing)
model = None
features = None
try:
    model = joblib.load("data/neo_risk_model.pkl")
except Exception as e:
    print("Warning: could not load model at data/neo_risk_model.pkl:", e)

try:
    features = joblib.load("data/model_features.pkl")
except Exception:
    # fallback to a reasonable default feature list used elsewhere in the repo
    features = [
        "H", "albedo", "diameter_sigma", "e", "a", "i",
        "absolute_magnitude_h", "estimated_diameter_min_km",
        "estimated_diameter_max_km", "relative_velocity_km_per_s", "miss_distance_km"
    ]

# Load dataset for visualization (optional)
data_path = "data/neo_risk_scores.csv"
if os.path.exists(data_path):
    try:
        neo_data = pd.read_csv(data_path)
    except Exception as e:
        print("Warning: failed to read data/neo_risk_scores.csv:", e)
        neo_data = pd.DataFrame()
else:
    neo_data = pd.DataFrame()

@app.route("/")
def home():
    top_risky = None

    # Prepare chart if dataset exists
    chart_path = None
    if not neo_data.empty:
        # try to pick sensible columns for plotting
        if "risk_score" in neo_data.columns:
            key = "risk_score"
        elif "risk" in neo_data.columns:
            key = "risk"
        else:
            key = None

        if key is not None:
            top_risky = neo_data.sort_values(key, ascending=False).head(10)
            plt.figure(figsize=(8, 5))
            # choose a label column if available
            if "full_name" in top_risky.columns:
                labels = top_risky["full_name"].astype(str)
            elif "name" in top_risky.columns:
                labels = top_risky["name"].astype(str)
            else:
                labels = top_risky.index.astype(str)

            # create static directory if needed
            static_dir = os.path.join(app.root_path, "static")
            os.makedirs(static_dir, exist_ok=True)
            chart_fname = "top_risks.png"
            chart_path_fs = os.path.join(static_dir, chart_fname)

# --- Better chart visualization ---
            plt.figure(figsize=(8, 5), facecolor="#0b0c10")
            ax = plt.gca()
            ax.set_facecolor("#1f2833")
            # Create readable category labels (if asteroid names missing)
            if "full_name" in top_risky.columns:
                labels = top_risky["full_name"].astype(str)
            elif "name" in top_risky.columns:
                labels = top_risky["name"].astype(str)
            else:
                labels = [f"NEO #{i+1}" for i in range(len(top_risky))]
            
            # Create color gradient (coolwarm colormap)
            import numpy as np
            colors = plt.cm.coolwarm(np.linspace(0.3, 1, len(top_risky)))

            bars = ax.barh(labels, top_risky[key], color=colors, edgecolor="white", linewidth=0.8)

            # Axis and title styling
            plt.xlabel("Risk Score (0–1)", color="#C5C6C7", fontsize=10)
            plt.ylabel("Asteroid", color="#C5C6C7", fontsize=10)
            plt.title("Top 10 Most Risky NEOs", color="#66FCF1", fontsize=13, weight="bold")

            ax.tick_params(colors="#C5C6C7")  # make tick labels light
            ax.spines['bottom'].set_color('#45A29E')
            ax.spines['left'].set_color('#45A29E')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.gca().invert_yaxis()  # highest risk on top
            # Add numeric labels beside bars
            for bar, score in zip(bars, top_risky[key]):
                plt.text(score + 0.01, bar.get_y() + bar.get_height()/2,f"{score:.2f}", va='center', fontsize=9, color='white', weight='bold')
            
            plt.xlim(0, 1.1)
            plt.tight_layout()
# --- End chart visualization ---

            try:
                plt.savefig(chart_path_fs)
                plt.close()
                # url_for requires an application context; build a relative URL
                chart_path = url_for('static', filename=chart_fname)
            except Exception as e:
                print("Warning: failed to save chart:", e)
                chart_path = None
        else:
            chart_path = None

    return render_template("index.html", chart=chart_path)

@app.route("/predict", methods=["POST"])
def predict():
    # compute risk_score in 0..1 (model output)
    risk_score = compute_risk(request.form)
    status = "Hazardous" if risk_score >= 0.5 else "Not Hazardous"
    bar_color = "#ff6b6b" if risk_score >= 0.5 else "#66fcf1"
    return render_template("index.html", risk=risk_score, status=status, bar_color=bar_color)


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def compute_risk(form) -> float:
    """Compute a risk score between 0 and 1.

    If a trained model is available it will be used. Otherwise a simple
    heuristic based on miss distance and relative velocity is used.
    """
    # Build a single-row DataFrame for prediction
    row = {}
    for f in features:
        v = form.get(f)
        row[f] = _safe_float(v, math.nan)

    try:
        import numpy as np
        X = pd.DataFrame([row])
    except Exception:
        X = None

    if model is not None and X is not None:
        try:
            # If model is a sklearn Pipeline it will accept the DataFrame
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X[features])[:, 1][0]
                return float(max(0.0, min(1.0, prob)))
            else:
                pred = model.predict(X[features])[0]
                p = float(pred)
                return float(max(0.0, min(1.0, p)))
        except Exception as e:
            print("Model prediction failed, falling back to heuristic:", e)

    # Fallback heuristic
    md = _safe_float(form.get("miss_distance_km"), None)
    rv = _safe_float(form.get("relative_velocity_km_per_s"), None)
    score = 0.0
    if md is not None and md > 0:
        # closer distances increase score; scale roughly so 1e6 km -> 0
        score += max(0.0, 1.0 - (md / 1e6))
    if rv is not None and rv > 0:
        # higher velocities increase score; 0..30 km/s mapped to 0..1
        score += min(1.0, rv / 30.0)

    # average contributing factors (0..2) -> 0..1
    score = score / 2.0
    score = max(0.0, min(1.0, score))
    return float(score)


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

