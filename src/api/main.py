from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.features.risk_features import calculate_features


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = PROJECT_ROOT / "app" / "transaction_history.json"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_fraud_model.pkl"


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="RiskLens AI Risk Manager",
    description="AI-powered transaction risk assessment system",
    version="0.2.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading RiskLens XGBoost model...")

model = joblib.load(MODEL_PATH)

print("RiskLens XGBoost model loaded.")
print("MODEL CLASSES:", model.classes_)
print(
    "MODEL FEATURE NAMES:",
    getattr(model, "feature_names_in_", "No feature names")
)

# =========================================================
# MODEL FEATURES
# =========================================================

FEATURES = [
    "TX_AMOUNT",
    "TX_AMOUNT_LOG",
    "HOUR",
    "DAY_OF_WEEK",
    "IS_WEEKEND",
    "CUSTOMER_TX_COUNT_BEFORE",
    "CUSTOMER_AVG_AMOUNT_BEFORE",
    "CUSTOMER_AMOUNT_STD_BEFORE",
    "CUSTOMER_AMOUNT_RATIO",
    "CUSTOMER_TX_COUNT_1H",
    "CUSTOMER_TX_COUNT_24H",
    "CUSTOMER_AMOUNT_24H",
    "TERMINAL_TX_COUNT_BEFORE",
    "TERMINAL_AVG_AMOUNT_BEFORE",
    "TERMINAL_AMOUNT_RATIO",
    "TERMINAL_TX_COUNT_1H",
    "TERMINAL_TX_COUNT_24H",
    "CUSTOMER_TERMINAL_TX_COUNT_BEFORE",
    "CUSTOMER_TERMINAL_SEEN_BEFORE",
    "CUSTOMER_AMOUNT_ZSCORE",
]


# =========================================================
# RISK THRESHOLDS
# =========================================================

LOW_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.60


# =========================================================
# TRANSACTION INPUT MODEL
# =========================================================

class Transaction(BaseModel):
    customer_id: int
    terminal_id: int
    amount: float
    timestamp: str


# =========================================================
# TRANSACTION HISTORY FUNCTIONS
# =========================================================

def load_transaction_history():
    if not HISTORY_PATH.exists():
        return []

    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_transaction_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


# =========================================================
# HOME / HEALTH CHECK
# =========================================================

@app.get("/")
def root():
    return {
        "name": "RiskLens AI Risk Manager",
        "status": "online",
        "version": "0.2.0",
    }


# =========================================================
# TRANSACTION RISK ASSESSMENT
# =========================================================

@app.post("/assess")
def assess_transaction(transaction: Transaction):

    # -----------------------------------------------------
    # 1. CALCULATE BEHAVIOURAL FEATURES
    # -----------------------------------------------------

    features = calculate_features(
        customer_id=transaction.customer_id,
        terminal_id=transaction.terminal_id,
        amount=transaction.amount,
        timestamp=transaction.timestamp,
    )

    # -----------------------------------------------------
    # 2. PREPARE MODEL INPUT
    # -----------------------------------------------------

    X = pd.DataFrame(
        [features],
        columns=FEATURES,
    )

    # -----------------------------------------------------
    # 3. GENERATE FRAUD PROBABILITY
    # -----------------------------------------------------

    risk_score = float(
        model.predict_proba(X)[0][1]
    )

    # -----------------------------------------------------
    # 4. DETERMINE RISK LEVEL
    # -----------------------------------------------------

    if risk_score < LOW_THRESHOLD:
        risk_level = "LOW"
        action = "APPROVE"

    elif risk_score < HIGH_THRESHOLD:
        risk_level = "MEDIUM"
        action = "AI_REVIEW"

    else:
        risk_level = "HIGH"
        action = "BLOCK"

    # -----------------------------------------------------
    # 5. GENERATE RISK EXPLANATIONS
    # -----------------------------------------------------

    reasons = []

    # Customer amount anomaly
    if features["CUSTOMER_AMOUNT_RATIO"] >= 3:
        reasons.append(
            f"Transaction amount is "
            f"{features['CUSTOMER_AMOUNT_RATIO']:.1f}x "
            f"the customer's typical amount."
        )

    # Statistical anomaly
    if features["CUSTOMER_AMOUNT_ZSCORE"] >= 3:
        reasons.append(
            f"Transaction amount is a significant "
            f"statistical outlier for this customer "
            f"(Z-score: "
            f"{features['CUSTOMER_AMOUNT_ZSCORE']:.1f})."
        )

    # Customer transaction frequency
    if features["CUSTOMER_TX_COUNT_1H"] >= 3:
        reasons.append(
            "Customer has unusually high transaction "
            "frequency within the last hour."
        )

    if features["CUSTOMER_TX_COUNT_24H"] >= 10:
        reasons.append(
            "Customer has unusually high transaction "
            "frequency within the last 24 hours."
        )

    # Terminal activity
    if features["TERMINAL_TX_COUNT_1H"] >= 3:
        reasons.append(
            "Terminal has multiple recent transactions."
        )

    # New customer-terminal relationship
    if features["CUSTOMER_TERMINAL_SEEN_BEFORE"] == 0:
        reasons.append(
            "Customer has not previously used this terminal."
        )

    # Terminal amount anomaly
    if features["TERMINAL_AMOUNT_RATIO"] >= 3:
        reasons.append(
            f"Transaction amount is "
            f"{features['TERMINAL_AMOUNT_RATIO']:.1f}x "
            f"the terminal's typical transaction amount."
        )

    # Fallback
    if not reasons:
        reasons.append(
            "No major behavioural anomalies detected."
        )

    # -----------------------------------------------------
    # 6. GENERATE SUMMARY
    # -----------------------------------------------------

    if risk_level == "HIGH":

        summary = (
            "Transaction shows multiple high-risk "
            "behavioural signals."
        )

    elif risk_level == "MEDIUM":

        summary = (
            "Transaction shows some behavioural anomalies "
            "and should be reviewed."
        )

    else:

        summary = (
            "Transaction appears consistent with "
            "normal behavioural patterns."
        )

    # -----------------------------------------------------
    # 7. BUILD RESPONSE
    # -----------------------------------------------------

    result = {
        "transaction": {
            "customer_id": transaction.customer_id,
            "terminal_id": transaction.terminal_id,
            "amount": transaction.amount,
            "timestamp": transaction.timestamp,
        },

        "risk_score": round(risk_score, 4),

        "risk_percentage": round(
            risk_score * 100,
            2,
        ),

        "risk_level": risk_level,

        "action": action,

        "explanation": {
            "summary": summary,
            "reasons": reasons,
        },

        "behavioural_signals": {

            "customer_transaction_count":
                features["CUSTOMER_TX_COUNT_BEFORE"],

            "customer_average_amount":
                round(
                    features["CUSTOMER_AVG_AMOUNT_BEFORE"],
                    2,
                ),

            "customer_amount_ratio":
                round(
                    features["CUSTOMER_AMOUNT_RATIO"],
                    2,
                ),

            "customer_transactions_1h":
                features["CUSTOMER_TX_COUNT_1H"],

            "customer_transactions_24h":
                features["CUSTOMER_TX_COUNT_24H"],

            "terminal_transaction_count":
                features["TERMINAL_TX_COUNT_BEFORE"],

            "terminal_average_amount":
                round(
                    features["TERMINAL_AVG_AMOUNT_BEFORE"],
                    2,
                ),

            "terminal_amount_ratio":
                round(
                    features["TERMINAL_AMOUNT_RATIO"],
                    2,
                ),

            "customer_amount_zscore":
                round(
                    features["CUSTOMER_AMOUNT_ZSCORE"],
                    2,
                ),

            "known_customer_terminal":
                bool(
                    features["CUSTOMER_TERMINAL_SEEN_BEFORE"]
                ),
        },

        "thresholds": {
            "low": LOW_THRESHOLD,
            "high": HIGH_THRESHOLD,
        },
    }

    # -----------------------------------------------------
    # 8. SAVE TRANSACTION TO HISTORY
    # -----------------------------------------------------

    history = load_transaction_history()

    history.append(result)

    save_transaction_history(history)

    # -----------------------------------------------------
    # 9. RETURN RESULT TO FRONTEND
    # -----------------------------------------------------

    return result


# =========================================================
# TRANSACTION HISTORY
# =========================================================

@app.get("/transactions")
def get_transactions():

    history = load_transaction_history()

    return {
        "count": len(history),
        "transactions": history,
    }