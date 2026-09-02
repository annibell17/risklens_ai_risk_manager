from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features_30d.csv"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TARGET = "TX_FRAUD"

# These are legitimate predictive features.
# We deliberately exclude TX_FRAUD and TX_FRAUD_SCENARIO.
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
    "TERMINAL_TX_COUNT_BEFORE",
    "TERMINAL_AVG_AMOUNT_BEFORE",
    "TERMINAL_AMOUNT_RATIO",
]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():

    print(f"Loading dataset:\n{INPUT_FILE}\n")

    df = pd.read_csv(INPUT_FILE)

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    df = df.sort_values("TX_DATETIME").reset_index(drop=True)

    print(f"Total transactions: {len(df):,}")

    return df


# ---------------------------------------------------------
# Time-based split
# ---------------------------------------------------------

def split_data(df):

    # First 70% = training
    # Last 30% = held-out test set

    split_index = int(len(df) * 0.70)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    print("\nTime-based split:")
    print(f"Training transactions: {len(train):,}")
    print(f"Test transactions:     {len(test):,}")

    print(
        f"\nTraining period:"
        f"\n  {train['TX_DATETIME'].min()}"
        f"\n  → {train['TX_DATETIME'].max()}"
    )

    print(
        f"\nTest period:"
        f"\n  {test['TX_DATETIME'].min()}"
        f"\n  → {test['TX_DATETIME'].max()}"
    )

    return train, test


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_model(X_train, y_train):

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    print("Training complete.")

    return model


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    # Default threshold for the first baseline.
    predictions = (probabilities >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    return probabilities


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    df = load_data()

    train, test = split_data(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\nFeatures used:")
    for feature in FEATURES:
        print(f"  - {feature}")

    model = train_model(
        X_train,
        y_train,
    )

    evaluate(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()