from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


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

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90
]

    # Hypothetical business costs
    fp_costs = [100, 500, 1000]
    fn_costs = [1000, 5000, 10000]

    print()
    print("=" * 110)
    print("COST-SENSITIVE THRESHOLD ANALYSIS")
    print("=" * 110)

    for fp_cost in fp_costs:

        for fn_cost in fn_costs:

            print()
            print(
                f"FALSE POSITIVE COST = ₹{fp_cost:,} | "
                f"FALSE NEGATIVE COST = ₹{fn_cost:,}"
            )

            print("-" * 90)
            print(
                f"{'Threshold':>10}"
                f"{'FP':>10}"
                f"{'FN':>10}"
                f"{'Recall':>12}"
                f"{'Precision':>12}"
                f"{'Total Cost':>16}"
            )

            best_threshold = None
            best_cost = float("inf")

            for threshold in thresholds:

                predictions = (probabilities >= threshold).astype(int)

                tn, fp, fn, tp = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1]
                ).ravel()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                total_cost = (
                    fp * fp_cost +
                    fn * fn_cost
                )

                print(
                    f"{threshold:>10.2f}"
                    f"{fp:>10,}"
                    f"{fn:>10,}"
                    f"{recall:>11.2%}"
                    f"{precision:>11.2%}"
                    f"₹{total_cost:>14,.0f}"
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_threshold = threshold

            print()
            print(
                f"BEST THRESHOLD: {best_threshold:.2f} "
                f"(Total cost: ₹{best_cost:,.0f})"
            )

    print()
    print("=" * 110)
    print("DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()