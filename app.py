
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

MODEL_PATH = BASE_DIR / "model" / "loan_risk_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
DATA_PATH = BASE_DIR / "dataset" / "loan_risk_dataset.csv"

model = None
metrics = {}
df = pd.DataFrame()

def load_assets():
    global model, metrics, df
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

def ensure_model():
    if model is None:
        raise RuntimeError("Model not found. Run: python train_model.py")

load_assets()

@app.context_processor
def inject_globals():
    accuracy = metrics.get("accuracy", 0) * 100
    return {
        "model_accuracy": round(accuracy, 1),
        "record_count": len(df) if not df.empty else 0,
        "feature_count": 8,
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        ensure_model()
        payload = request.get_json(silent=True) or request.form

        age = float(payload.get("age"))
        income = float(payload.get("income"))
        loan_amount = float(payload.get("loan_amount"))
        credit_score = float(payload.get("credit_score"))
        employment = str(payload.get("employment"))
        loan_term = int(payload.get("loan_term"))
        previous_history = str(payload.get("previous_history"))
        dti = float(payload.get("dti"))

        errors = []
        if not 18 <= age <= 100: errors.append("Age must be between 18 and 100.")
        if income <= 0: errors.append("Annual income must be greater than 0.")
        if loan_amount <= 0: errors.append("Loan amount must be greater than 0.")
        if not 300 <= credit_score <= 850: errors.append("Credit score must be between 300 and 850.")
        if loan_term not in [12, 18, 24, 36, 48, 60]:
            errors.append("Select a supported loan term.")
        if not 0 < dti <= 1: errors.append("Debt-to-income ratio must be between 0 and 1.")

        if errors:
            return jsonify({"ok": False, "errors": errors}), 400

        sample = pd.DataFrame([{
            "Age": age,
            "Annual Income": income,
            "Loan Amount": loan_amount,
            "Credit Score": credit_score,
            "Employment Status": employment,
            "Loan Term": loan_term,
            "Previous Loan History": previous_history,
            "Debt to Income Ratio": dti,
        }])

        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        classes = list(model.classes_)
        prob_map = {cls: float(prob) * 100 for cls, prob in zip(classes, probabilities)}

        # Helpful human-readable indicators derived from input, while the risk category
        # and probabilities remain ML-model outputs.
        credit_impact = "High" if credit_score >= 700 else ("Medium" if credit_score >= 600 else "Low")
        income_stability = "Good" if income >= 600000 else ("Moderate" if income >= 350000 else "Needs Review")
        ratio = loan_amount / income
        loan_risk = "Low" if ratio < 0.35 else ("Medium" if ratio < 0.60 else "High")
        assessment = {
            "Low Risk": "Favorable",
            "Medium Risk": "Review Recommended",
            "High Risk": "Higher Risk"
        }.get(prediction, "Review Recommended")

        return jsonify({
            "ok": True,
            "risk": prediction,
            "probabilities": {
                "Low Risk": round(prob_map.get("Low Risk", 0), 1),
                "Medium Risk": round(prob_map.get("Medium Risk", 0), 1),
                "High Risk": round(prob_map.get("High Risk", 0), 1),
            },
            "factors": {
                "credit": credit_impact,
                "income": income_stability,
                "loan_amount": loan_risk,
                "assessment": assessment
            }
        })
    except Exception as exc:
        return jsonify({"ok": False, "errors": [f"Prediction error: {exc}"]}), 500

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/about")
def about():
    return render_template("about.html", metrics=metrics)

@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
