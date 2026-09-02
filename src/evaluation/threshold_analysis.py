from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


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
    "TERMINAL_TX_COUNT_BEFORE",
    "TERMINAL_AVG_AMOUNT_BEFORE",
    "TERMINAL_AMOUNT_RATIO",
]

TARGET = "TX_FRAUD"


def main():

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

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative / positive,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        f"\n{'Threshold':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1':<14}"
        f"{'Alerts':<12}"
    )

    print("-" * 66)

    for threshold in [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        alerts = predictions.sum()

        print(
            f"{threshold:<12.2f}"
            f"{precision:<14.4f}"
            f"{recall:<14.4f}"
            f"{f1:<14.4f}"
            f"{alerts:<12,}"
        )


if __name__ == "__main__":
    main()