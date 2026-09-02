from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features_30d.csv"
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
# LOAD HISTORICAL DATA
# =========================================================

print("Loading historical transaction data...")

df = pd.read_csv(DATA_PATH)

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

df = df.sort_values("TX_DATETIME").reset_index(drop=True)

print(f"Loaded {len(df):,} historical transactions.")


# =========================================================
# FEATURE CALCULATOR
# =========================================================

def calculate_features(
    customer_id: int,
    terminal_id: int,
    amount: float,
    timestamp: str,
):
    """
    Calculate behavioural features for a new transaction
    using transactions that occurred before it.
    """

    timestamp = pd.to_datetime(timestamp)

    # -----------------------------------------------------
    # Basic temporal features
    # -----------------------------------------------------

    hour = timestamp.hour + timestamp.minute / 60

    day_of_week = timestamp.dayofweek

    is_weekend = int(day_of_week >= 5)

    amount_log = np.log1p(amount)


    # -----------------------------------------------------
    # Historical transactions BEFORE this transaction
    # -----------------------------------------------------

    history = df[df["TX_DATETIME"] < timestamp]


    # -----------------------------------------------------
    # Customer history
    # -----------------------------------------------------

    customer_history = history[
        history["CUSTOMER_ID"] == customer_id
    ]

    customer_count = len(customer_history)


    if customer_count > 0:

        customer_avg = (
            customer_history["TX_AMOUNT"]
            .mean()
        )

        customer_std = (
            customer_history["TX_AMOUNT"]
            .std()
        )

        if pd.isna(customer_std):
            customer_std = 0.0

    else:

        customer_avg = 0.0
        customer_std = 0.0


    # Amount relative to customer's normal spending

    if customer_avg > 0:

        customer_amount_ratio = (
            amount / customer_avg
        )

    else:

        customer_amount_ratio = 0.0


    # -----------------------------------------------------
    # Customer 1-hour activity
    # -----------------------------------------------------

    one_hour_start = (
        timestamp - pd.Timedelta(hours=1)
    )

    customer_1h = customer_history[
        customer_history["TX_DATETIME"] >= one_hour_start
    ]

    customer_tx_count_1h = len(customer_1h)


    # -----------------------------------------------------
    # Customer 24-hour activity
    # -----------------------------------------------------

    twenty_four_hour_start = (
        timestamp - pd.Timedelta(hours=24)
    )

    customer_24h = customer_history[
        customer_history["TX_DATETIME"] >= twenty_four_hour_start
    ]

    customer_tx_count_24h = len(customer_24h)

    customer_amount_24h = (
        customer_24h["TX_AMOUNT"].sum()
    )


    # -----------------------------------------------------
    # Customer amount Z-score
    # -----------------------------------------------------

    if customer_std > 0:

        customer_amount_zscore = (
            (amount - customer_avg)
            / customer_std
        )

    else:

        customer_amount_zscore = 0.0


    # -----------------------------------------------------
    # Terminal history
    # -----------------------------------------------------

    terminal_history = history[
        history["TERMINAL_ID"] == terminal_id
    ]

    terminal_count = len(terminal_history)


    if terminal_count > 0:

        terminal_avg = (
            terminal_history["TX_AMOUNT"]
            .mean()
        )

    else:

        terminal_avg = 0.0


    # Amount relative to terminal average

    if terminal_avg > 0:

        terminal_amount_ratio = (
            amount / terminal_avg
        )

    else:

        terminal_amount_ratio = 0.0


    # -----------------------------------------------------
    # Terminal activity
    # -----------------------------------------------------

    terminal_1h = terminal_history[
        terminal_history["TX_DATETIME"] >= one_hour_start
    ]

    terminal_tx_count_1h = len(terminal_1h)


    terminal_24h = terminal_history[
        terminal_history["TX_DATETIME"] >= twenty_four_hour_start
    ]

    terminal_tx_count_24h = len(terminal_24h)


    # -----------------------------------------------------
    # Customer-terminal relationship
    # -----------------------------------------------------

    customer_terminal_history = history[
        (history["CUSTOMER_ID"] == customer_id)
        &
        (history["TERMINAL_ID"] == terminal_id)
    ]

    customer_terminal_count = (
        len(customer_terminal_history)
    )

    customer_terminal_seen = int(
        customer_terminal_count > 0
    )


    # =====================================================
    # RESULT
    # =====================================================

    features = {

        "TX_AMOUNT": amount,

        "TX_AMOUNT_LOG": amount_log,

        "HOUR": hour,

        "DAY_OF_WEEK": day_of_week,

        "IS_WEEKEND": is_weekend,

        "CUSTOMER_TX_COUNT_BEFORE":
            customer_count,

        "CUSTOMER_AVG_AMOUNT_BEFORE":
            customer_avg,

        "CUSTOMER_AMOUNT_STD_BEFORE":
            customer_std,

        "CUSTOMER_AMOUNT_RATIO":
            customer_amount_ratio,

        "CUSTOMER_TX_COUNT_1H":
            customer_tx_count_1h,

        "CUSTOMER_TX_COUNT_24H":
            customer_tx_count_24h,

        "CUSTOMER_AMOUNT_24H":
            customer_amount_24h,

        "TERMINAL_TX_COUNT_BEFORE":
            terminal_count,

        "TERMINAL_AVG_AMOUNT_BEFORE":
            terminal_avg,

        "TERMINAL_AMOUNT_RATIO":
            terminal_amount_ratio,

        "TERMINAL_TX_COUNT_1H":
            terminal_tx_count_1h,

        "TERMINAL_TX_COUNT_24H":
            terminal_tx_count_24h,

        "CUSTOMER_TERMINAL_TX_COUNT_BEFORE":
            customer_terminal_count,

        "CUSTOMER_TERMINAL_SEEN_BEFORE":
            customer_terminal_seen,

        "CUSTOMER_AMOUNT_ZSCORE":
            customer_amount_zscore,
    }

    return features


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("\nTesting RiskLens feature calculator...\n")

    test_features = calculate_features(
        customer_id=596,
        terminal_id=3156,
        amount=250.00,
        timestamp="2018-04-15 14:32:00",
    )

    print("Calculated features:")

    for name in FEATURES:

        print(
            f"{name:<40} "
            f"{test_features[name]}"
        )