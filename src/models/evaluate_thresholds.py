from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "transactions_features_30d.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_fraud_model.pkl"

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
    df = pd.read_csv(DATA_FILE)

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)

    # Same chronological split used during training
    split_index = int(len(df) * 0.70)

    test = df.iloc[split_index:]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(f"Test transactions: {len(test):,}")
    print(f"Test fraud transactions: {y_test.sum():,}")
    print()

    print("Loading XGBoost model...")
    model = joblib.load(MODEL_FILE)

    print("Generating probabilities...")
    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        false_positive_rate = fp / (fp + tn)

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "False Positive Rate": false_positive_rate,
            "True Positives": tp,
            "False Positives": fp,
            "False Negatives": fn,
            "True Negatives": tn,
        })

    results_df = pd.DataFrame(results)

    print()
    print("=" * 110)
    print("THRESHOLD ANALYSIS")
    print("=" * 110)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
                "False Positive Rate": "{:.4%}".format,
            }
        )
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLDS")
    print("=" * 70)

    best_f1 = results_df.loc[results_df["F1"].idxmax()]
    best_precision = results_df.loc[results_df["Precision"].idxmax()]
    best_recall = results_df.loc[results_df["Recall"].idxmax()]

    print(
        f"Best F1:        threshold={best_f1['Threshold']:.2f}, "
        f"F1={best_f1['F1']:.4f}"
    )

    print(
        f"Best Precision: threshold={best_precision['Threshold']:.2f}, "
        f"Precision={best_precision['Precision']:.4f}"
    )

    print(
        f"Best Recall:    threshold={best_recall['Threshold']:.2f}, "
        f"Recall={best_recall['Recall']:.4f}"
    )

    output_file = PROJECT_ROOT / "data" / "processed" / "threshold_analysis.csv"
    results_df.to_csv(output_file, index=False)

    print()
    print(f"Saved results to:")
    print(output_file)


if __name__ == "__main__":
    main()
    