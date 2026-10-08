
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
CHART_DIR = BASE_DIR / "static" / "charts"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N = 4200
rng = np.random.default_rng(RANDOM_STATE)

# Reproducible realistic synthetic loan-risk dataset.
age = rng.integers(21, 66, N)
income = np.clip(rng.lognormal(mean=np.log(650000), sigma=0.48, size=N), 180000, 2200000).round(-2)
loan_amount = np.clip(income * rng.uniform(0.18, 0.85, N) + rng.normal(0, 65000, N), 50000, 1500000).round(-2)
credit_score = np.clip(
    760
    + (income - income.mean()) / income.std() * 16
    - (loan_amount / np.maximum(income, 1)) * 55
    + rng.normal(0, 58, N),
    300, 850
).round().astype(int)

employment = rng.choice(
    ["Salaried", "Self-Employed", "Business", "Other"],
    size=N,
    p=[0.48, 0.22, 0.20, 0.10]
)
loan_term = rng.choice([12, 18, 24, 36, 48, 60], size=N, p=[0.05, 0.07, 0.14, 0.36, 0.23, 0.15])
previous_history = rng.choice(["Yes", "No"], size=N, p=[0.62, 0.38])

dti = np.clip(
    0.18
    + (loan_amount / np.maximum(income, 1)) * 0.48
    + rng.normal(0, 0.10, N),
    0.05, 0.85
).round(2)

# Latent risk score. Noise prevents the classifier from simply memorizing the label.
score = (
    0.45 * (700 - credit_score) / 100
    + 1.35 * dti
    + 0.55 * (loan_amount / np.maximum(income, 1))
    + 0.16 * (loan_term >= 48)
    + 0.16 * (previous_history == "Yes")
    + 0.12 * (employment == "Other")
    + 0.08 * (age < 25)
    + rng.normal(0, 0.08, N)
)

# Quantile thresholds give balanced, useful classes.
q1, q2 = np.quantile(score, [0.68, 0.90])
risk = np.where(score <= q1, "Low Risk", np.where(score <= q2, "Medium Risk", "High Risk"))

df = pd.DataFrame({
    "Age": age,
    "Annual Income": income.astype(int),
    "Loan Amount": loan_amount.astype(int),
    "Credit Score": credit_score,
    "Employment Status": employment,
    "Loan Term": loan_term,
    "Previous Loan History": previous_history,
    "Debt to Income Ratio": dti,
    "Loan Risk": risk,
})
csv_path = DATA_DIR / "loan_risk_dataset.csv"
df.to_csv(csv_path, index=False)

X = df.drop(columns=["Loan Risk"])
y = df["Loan Risk"]

categorical = ["Employment Status", "Previous Loan History"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, pred)

joblib.dump(pipeline, MODEL_DIR / "loan_risk_model.pkl")

report = classification_report(y_test, pred, output_dict=True)
metrics = {
    "accuracy": float(accuracy),
    "test_records": int(len(y_test)),
    "total_records": int(len(df)),
    "features": int(len(X.columns)),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(
        y_test, pred, labels=["Low Risk", "Medium Risk", "High Risk"]
    ).tolist()
}
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# Create charts from the actual dataset.
plt.rcParams.update({"figure.dpi": 120})

# 1. Risk distribution
counts = df["Loan Risk"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"])
fig, ax = plt.subplots(figsize=(5.2, 3.5))
ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90)
ax.set_title("Loan Risk Distribution", fontweight="bold")
fig.tight_layout()
fig.savefig(CHART_DIR / "risk_distribution.png", bbox_inches="tight")
plt.close(fig)

# 2. Income vs Loan Amount
fig, ax = plt.subplots(figsize=(5.2, 3.5))
for label in ["Low Risk", "Medium Risk", "High Risk"]:
    part = df[df["Loan Risk"] == label]
    ax.scatter(part["Annual Income"], part["Loan Amount"], s=9, alpha=0.35, label=label)
ax.set_title("Income vs Loan Amount", fontweight="bold")
ax.set_xlabel("Annual Income (₹)")
ax.set_ylabel("Loan Amount (₹)")
ax.ticklabel_format(style="plain", axis="x")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(CHART_DIR / "income_vs_loan.png", bbox_inches="tight")
plt.close(fig)

# 3. Credit score distribution
fig, ax = plt.subplots(figsize=(5.2, 3.5))
ax.hist(df["Credit Score"], bins=24, edgecolor="white")
ax.set_title("Credit Score Distribution", fontweight="bold")
ax.set_xlabel("Credit Score")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(CHART_DIR / "credit_distribution.png", bbox_inches="tight")
plt.close(fig)

# 4. Risk by employment status
pivot = pd.crosstab(df["Employment Status"], df["Loan Risk"]).reindex(
    columns=["Low Risk", "Medium Risk", "High Risk"], fill_value=0
)
fig, ax = plt.subplots(figsize=(5.2, 3.5))
pivot.plot(kind="bar", ax=ax)
ax.set_title("Risk by Employment Status", fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(CHART_DIR / "risk_by_employment.png", bbox_inches="tight")
plt.close(fig)

print(f"Dataset: {csv_path}")
print(f"Accuracy: {accuracy:.4f}")
print("Model and charts generated successfully.")
