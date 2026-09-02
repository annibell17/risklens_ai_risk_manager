from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features_30d.csv"
)

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


TARGET = "TX_FRAUD"


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)

    split_index = int(len(df) * 0.70)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Calculate imbalance ratio.
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    print(f"\nLegitimate training transactions: {negative:,}")
    print(f"Fraud training transactions:       {positive:,}")
    print(f"Scale positive weight:             {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")

    model.fit(X_train, y_train)

    print("Training complete.")

    probabilities = model.predict_proba(X_test)[:, 1]

    # Initial threshold.
    predictions = (probabilities >= 0.5).astype(int)

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

    print("\n" + "=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)

    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nFeature importance:")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    
    print(importance.to_string())

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_fraud_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")


if __name__ == "__main__":
    main()
