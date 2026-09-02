from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


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


# ---------------------------------------------------------
# Business assumptions
# ---------------------------------------------------------

# These are DEMO assumptions, not Razorpay's actual costs.

FRAUD_MISS_COST = 1000
FALSE_POSITIVE_COST = 20


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

    print("\n" + "=" * 80)
    print("COST-SENSITIVE RISK ANALYSIS")
    print("=" * 80)

    print(f"\nAssumed cost of missed fraud: ₹{FRAUD_MISS_COST:,}")
    print(f"Assumed cost of false positive: ₹{FALSE_POSITIVE_COST:,}")

    print(
        f"\n{'Threshold':<12}"
        f"{'FP':<10}"
        f"{'FN':<10}"
        f"{'Precision':<14}"
        f"{'Recall':<12}"
        f"{'Total Cost':<15}"
    )

    print("-" * 75)

    results = []

    for threshold in [
        0.30,
        0.40,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        false_positives = (
            ((predictions == 1) & (y_test == 0))
            .sum()
        )

        false_negatives = (
            ((predictions == 0) & (y_test == 1))
            .sum()
        )

        true_positives = (
            ((predictions == 1) & (y_test == 1))
            .sum()
        )

        precision = (
            true_positives
            / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )

        recall = (
            true_positives
            / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        total_cost = (
            false_positives * FALSE_POSITIVE_COST
            + false_negatives * FRAUD_MISS_COST
        )

        results.append(
            {
                "threshold": threshold,
                "fp": false_positives,
                "fn": false_negatives,
                "precision": precision,
                "recall": recall,
                "cost": total_cost,
            }
        )

        print(
            f"{threshold:<12.2f}"
            f"{false_positives:<10,}"
            f"{false_negatives:<10,}"
            f"{precision:<14.4f}"
            f"{recall:<12.4f}"
            f"₹{total_cost:<14,}"
        )

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["cost"].idxmin()
    ]

    print("\n" + "=" * 80)
    print("LOWEST-COST THRESHOLD")
    print("=" * 80)

    print(f"\nThreshold : {best['threshold']:.2f}")
    print(f"Precision : {best['precision']:.4f}")
    print(f"Recall    : {best['recall']:.4f}")
    print(f"FP        : {int(best['fp']):,}")
    print(f"FN        : {int(best['fn']):,}")
    print(f"Cost      : ₹{int(best['cost']):,}")


if __name__ == "__main__":
    main()