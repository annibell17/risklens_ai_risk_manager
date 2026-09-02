# RiskLens — AI Risk Manager

An AI-powered transaction risk assessment system designed to detect potentially fraudulent payment transactions and provide an immediate, explainable risk decision.

RiskLens combines machine learning with behavioural transaction signals to evaluate each transaction and classify it into a risk tier:

- 🟢 LOW — Approve
- 🟡 MEDIUM — Step-up verification / review
- 🔴 HIGH — Block / manual review

## Problem

Payment fraud detection is not simply a matter of identifying unusually large transactions. Fraudulent behaviour can also appear through changes in customer spending patterns, transaction velocity, terminal activity, and previously unseen customer-terminal relationships.

RiskLens addresses this by combining transaction-level information with historical behavioural features to produce a real-time risk score and an actionable decision.

## Dataset

RiskLens was trained using the simulated transaction dataset from the Fraud Detection Handbook.

The dataset contains chronological payment transactions with transaction, customer, terminal, amount, timestamp, and fraud-label information.

A chronological train/test split was used to preserve the temporal nature of the fraud detection problem and avoid evaluating the model on randomly mixed future and past transactions.

## Feature Engineering

Rather than relying only on the transaction amount, RiskLens derives behavioural features from historical transaction activity.

Key features include:

- Transaction amount and log-transformed amount
- Transaction hour, day of week, and weekend indicator
- Customer transaction count and historical average amount
- Customer amount ratio and amount z-score
- Customer transaction velocity over the previous 1 hour and 24 hours
- Customer transaction amount over the previous 24 hours
- Terminal transaction count and historical average amount
- Terminal amount ratio and transaction velocity
- Customer-terminal interaction history
- Whether the customer-terminal combination has been observed previously

### Temporal Integrity

Behavioural features are calculated using only information available **before the transaction being assessed**.

Historical aggregations use strict temporal filtering, ensuring that the current transaction does not contribute to its own behavioural features.

This prevents future information and same-transaction information from leaking into the model during training or inference.

The fraud label is used only as the target variable and is not used as an input feature.

## Model

RiskLens uses an XGBoost binary classification model to estimate the probability that a transaction is fraudulent.

The model was trained using the following configuration:

- Algorithm: XGBoost
- Number of estimators: 300
- Maximum tree depth: 6
- Learning rate: 0.05
- Subsample: 0.8
- Column subsampling: 0.8
- Objective: Binary logistic classification
- Evaluation metric during training: PR-AUC
- Class imbalance handling: `scale_pos_weight`

The model uses 20 transaction and behavioural features. Fraud labels and fraud-scenario identifiers are not used as input features.

## Evaluation

The dataset was split chronologically, with earlier transactions used for training and later transactions reserved as a held-out test set.

The test set contains 86,419 transactions, including 763 fraudulent transactions.

At the baseline classification threshold of 0.50, RiskLens achieved:

| Metric | Result |
|---|---:|
| Precision | 13.99% |
| Recall | 34.99% |
| F1 Score | 19.99% |
| ROC-AUC | 0.6871 |
| PR-AUC | 0.3153 |

### Confusion Matrix

At the 0.50 threshold:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 84,015 | 1,641 |
| Actual Fraud | 496 | 267 |

These results are reported on a chronological held-out test set rather than a randomly shuffled split.

Because fraud detection is highly imbalanced, accuracy is not treated as the primary performance measure. Precision, recall, F1, PR-AUC, and false-positive rate provide a more informative view of model performance.

## Operating Threshold

Model evaluation and product policy are treated separately.

The baseline model metrics above use a threshold of 0.50 for reproducible model evaluation. For the deployed risk policy, thresholds were evaluated using a cost-sensitive analysis across multiple illustrative false-positive and false-negative cost assumptions.

The selected operating point is:

- **LOW:** risk score < 0.40 → Approve
- **MEDIUM:** 0.40 ≤ risk score < 0.70 → Step-up verification / review
- **HIGH:** risk score ≥ 0.70 → Block / manual review

At a 0.70 classification threshold on the held-out test set:

| Metric | Result |
|---|---:|
| Precision | 59.54% |
| Recall | 30.67% |
| False-positive rate | 0.1856% |
| True positives | 234 |
| False positives | 159 |
| False negatives | 529 |
| True negatives | 85,497 |

The 0.70 threshold was the cost-minimizing threshold under 4 of the 9 illustrative cost scenarios tested, including the middle assumption of ₹500 per false positive and ₹5,000 per false negative.

These costs are illustrative rather than production business costs. In a real payment system, threshold selection would be calibrated using actual fraud losses, customer-friction costs, operational review capacity, and the organization's risk appetite.

## System Architecture

RiskLens follows a transaction-to-decision pipeline:

```text
                    ┌─────────────────────┐
                    │  Transaction Input  │
                    │                     │
                    │ Customer ID         │
                    │ Terminal ID         │
                    │ Amount              │
                    │ Timestamp           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │                     │
                    │ Transaction features│
                    │ Customer behaviour  │
                    │ Terminal behaviour  │
                    │ Velocity signals    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   XGBoost Model     │
                    │                     │
                    │ Fraud probability   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Policy       │
                    │                     │
                    │ < 0.40  → LOW       │
                    │ 0.40–0.70 → MEDIUM  │
                    │ ≥ 0.70 → HIGH       │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
     ┌─────────────────┐              ┌──────────────────┐
     │ Action          │              │ Explanation      │
     │                 │              │                  │
     │ APPROVE         │              │ Behavioural      │
     │ STEP-UP / REVIEW│              │ risk signals     │
     │ BLOCK / REVIEW  │              │ contributing     │
     └─────────────────┘              └──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Transaction History │
                    └─────────────────────┘

                    Backend

## Project Structure

```text
risklens_ai_risk_manager/
│
├── src/
│   └── api/
│       └── main.py           # FastAPI application
│
├── app/
│   └── transaction_history.json
│
├── data/
│   ├── raw/                  # Raw simulated transaction data
│   └── processed/            # Engineered transaction features
│
├── models/
│   └── xgboost_fraud_model.pkl
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── requirements.txt
└── README.md
The backend is implemented using FastAPI and provides an API endpoint for transaction risk assessment.

The assessment pipeline:

Receives transaction details from the frontend.
Generates behavioural and transaction-level features using historical data available before the transaction.
Passes the feature vector to the trained XGBoost model.
Converts the model probability into a risk tier using the configured operating thresholds.
Returns the risk score, risk level, recommended action, behavioural signals, and explanation.
Stores the assessment in transaction history.
Frontend

The frontend provides two primary views:

Risk Assessment — enter a transaction and receive an immediate risk assessment.
Transactions — view previously assessed transactions and their outcomes.

The frontend communicates with the FastAPI backend through HTTP requests.

Explainable Risk Assessment

RiskLens does not return a risk score alone. Each assessment is accompanied by behavioural signals that help explain why a transaction may be considered unusual.

Examples include:

Customer amount ratio — how large the current transaction is relative to the customer's historical spending behaviour.
Customer amount z-score — how far the transaction amount deviates from the customer's historical amount distribution.
Customer transaction velocity — activity by the customer within recent time windows.
Terminal transaction velocity — recent activity associated with the payment terminal.
Customer-terminal history — whether the customer has previously used the terminal.

These signals provide contextual evidence alongside the model's probability estimate, allowing a reviewer to understand the behavioural factors associated with a risk decision.

The explanation layer is intended to support human review and does not claim that any individual behavioural signal independently proves fraud.

How to Run
Prerequisites
Python 3.10+
pip
A modern web browser
### Backend

The backend is implemented using FastAPI and provides an API endpoint for transaction risk assessment.

The assessment pipeline:

1. Receives transaction details from the frontend.
2. Generates behavioural and transaction-level features using historical data available before the transaction.
3. Passes the feature vector to the trained XGBoost model.
4. Converts the model probability into a risk tier using the configured operating thresholds.
5. Returns the risk score, risk level, recommended action, behavioural signals, and explanation.
6. Stores the assessment in transaction history.

### Frontend

The frontend provides two primary views:

- **Risk Assessment** — enter a transaction and receive an immediate risk assessment.
- **Transactions** — view previously assessed transactions and their outcomes.

The frontend communicates with the FastAPI backend through HTTP requests.

## Explainable Risk Assessment

RiskLens does not return a risk score alone. Each assessment is accompanied by behavioural signals that help explain why a transaction may be considered unusual.

Examples include:

- **Customer amount ratio** — how large the current transaction is relative to the customer's historical spending behaviour.
- **Customer amount z-score** — how far the transaction amount deviates from the customer's historical amount distribution.
- **Customer transaction velocity** — activity by the customer within recent time windows.
- **Terminal transaction velocity** — recent activity associated with the payment terminal.
- **Customer-terminal history** — whether the customer has previously used the terminal.

These signals provide contextual evidence alongside the model's probability estimate, allowing a reviewer to understand the behavioural factors associated with a risk decision.

The explanation layer is intended to support human review and does not claim that any individual behavioural signal independently proves fraud.

## How to Run

### Prerequisites

- Python 3.10+
- pip
- A modern web browser

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd risklens_ai_risk_manager