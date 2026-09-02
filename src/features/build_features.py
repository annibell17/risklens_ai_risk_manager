from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_30d.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features_30d.csv"
)


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    df = (
        df.sort_values(
            ["TX_DATETIME", "TRANSACTION_ID"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    df["HOUR"] = (
        df["TX_DATETIME"].dt.hour
        + df["TX_DATETIME"].dt.minute / 60
    )

    df["DAY_OF_WEEK"] = (
        df["TX_DATETIME"].dt.dayofweek
    )

    df["IS_WEEKEND"] = (
        df["DAY_OF_WEEK"] >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Amount features
    # -----------------------------------------------------

    df["TX_AMOUNT_LOG"] = (
        np.log1p(df["TX_AMOUNT"])
    )

    # -----------------------------------------------------
    # Customer historical behaviour
    #
    # IMPORTANT:
    # Only transactions with a STRICTLY EARLIER timestamp
    # are allowed to contribute to historical statistics.
    # -----------------------------------------------------

    df["CUSTOMER_TX_COUNT_BEFORE"] = 0
    df["CUSTOMER_AVG_AMOUNT_BEFORE"] = 0.0
    df["CUSTOMER_AMOUNT_STD_BEFORE"] = 0.0

    for customer_id, group in df.groupby(
        "CUSTOMER_ID",
        sort=False
    ):

        indices = group.index.to_numpy()
        times = (
            group["TX_DATETIME"]
            .astype("int64")
            .to_numpy()
        )
        amounts = (
            group["TX_AMOUNT"]
            .to_numpy(dtype=float)
        )

        cumulative_sum = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        cumulative_sum_sq = np.concatenate(
            ([0.0], np.cumsum(amounts ** 2))
        )

        for i, current_time in enumerate(times):

            # First position at the current timestamp.
            # Everything before this position is strictly earlier.
            right = np.searchsorted(
                times,
                current_time,
                side="left"
            )

            count = right

            if count > 0:

                total = cumulative_sum[right]

                mean = total / count

                variance = (
                    cumulative_sum_sq[right] / count
                    - mean ** 2
                )

                variance = max(variance, 0.0)

                std = np.sqrt(variance)

            else:

                mean = 0.0
                std = 0.0

            idx = indices[i]

            df.at[
                idx,
                "CUSTOMER_TX_COUNT_BEFORE"
            ] = count

            df.at[
                idx,
                "CUSTOMER_AVG_AMOUNT_BEFORE"
            ] = mean

            df.at[
                idx,
                "CUSTOMER_AMOUNT_STD_BEFORE"
            ] = std

    # Amount relative to customer's historical average
    df["CUSTOMER_AMOUNT_RATIO"] = (
        df["TX_AMOUNT"]
        /
        df["CUSTOMER_AVG_AMOUNT_BEFORE"]
        .replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Terminal historical behaviour
    # -----------------------------------------------------

    df["TERMINAL_TX_COUNT_BEFORE"] = 0
    df["TERMINAL_AVG_AMOUNT_BEFORE"] = 0.0

    for terminal_id, group in df.groupby(
        "TERMINAL_ID",
        sort=False
    ):

        indices = group.index.to_numpy()

        times = (
            group["TX_DATETIME"]
            .astype("int64")
            .to_numpy()
        )

        amounts = (
            group["TX_AMOUNT"]
            .to_numpy(dtype=float)
        )

        cumulative_sum = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        for i, current_time in enumerate(times):

            right = np.searchsorted(
                times,
                current_time,
                side="left"
            )

            count = right

            if count > 0:
                mean = (
                    cumulative_sum[right]
                    / count
                )
            else:
                mean = 0.0

            idx = indices[i]

            df.at[
                idx,
                "TERMINAL_TX_COUNT_BEFORE"
            ] = count

            df.at[
                idx,
                "TERMINAL_AVG_AMOUNT_BEFORE"
            ] = mean

    df["TERMINAL_AMOUNT_RATIO"] = (
        df["TX_AMOUNT"]
        /
        df["TERMINAL_AVG_AMOUNT_BEFORE"]
        .replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Customer velocity
    #
    # Counts transactions from the same customer in:
    #
    #   [current_time - window, current_time)
    #
    # The right boundary is EXCLUSIVE, meaning transactions
    # at exactly the same timestamp are not counted.
    # -----------------------------------------------------

    df["CUSTOMER_TX_COUNT_1H"] = 0
    df["CUSTOMER_TX_COUNT_24H"] = 0
    df["CUSTOMER_AMOUNT_24H"] = 0.0

    for customer_id, group in df.groupby(
        "CUSTOMER_ID",
        sort=False
    ):

        indices = group.index.to_numpy()

        times = (
            group["TX_DATETIME"]
            .astype("int64")
            .to_numpy()
        )

        amounts = (
            group["TX_AMOUNT"]
            .to_numpy(dtype=float)
        )

        cumulative_sum = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        one_hour = pd.Timedelta(
            hours=1
        ).value

        twenty_four_hours = pd.Timedelta(
            hours=24
        ).value

        for i, current_time in enumerate(times):

            # Strictly earlier than current timestamp
            right = np.searchsorted(
                times,
                current_time,
                side="left"
            )

            left_1h = np.searchsorted(
                times,
                current_time - one_hour,
                side="left"
            )

            left_24h = np.searchsorted(
                times,
                current_time - twenty_four_hours,
                side="left"
            )

            count_1h = max(
                right - left_1h,
                0
            )

            count_24h = max(
                right - left_24h,
                0
            )

            amount_24h = (
                cumulative_sum[right]
                - cumulative_sum[left_24h]
            )

            idx = indices[i]

            df.at[
                idx,
                "CUSTOMER_TX_COUNT_1H"
            ] = count_1h

            df.at[
                idx,
                "CUSTOMER_TX_COUNT_24H"
            ] = count_24h

            df.at[
                idx,
                "CUSTOMER_AMOUNT_24H"
            ] = amount_24h

    # -----------------------------------------------------
    # Terminal velocity
    # -----------------------------------------------------

    df["TERMINAL_TX_COUNT_1H"] = 0
    df["TERMINAL_TX_COUNT_24H"] = 0

    for terminal_id, group in df.groupby(
        "TERMINAL_ID",
        sort=False
    ):

        indices = group.index.to_numpy()

        times = (
            group["TX_DATETIME"]
            .astype("int64")
            .to_numpy()
        )

        one_hour = pd.Timedelta(
            hours=1
        ).value

        twenty_four_hours = pd.Timedelta(
            hours=24
        ).value

        for i, current_time in enumerate(times):

            right = np.searchsorted(
                times,
                current_time,
                side="left"
            )

            left_1h = np.searchsorted(
                times,
                current_time - one_hour,
                side="left"
            )

            left_24h = np.searchsorted(
                times,
                current_time - twenty_four_hours,
                side="left"
            )

            count_1h = max(
                right - left_1h,
                0
            )

            count_24h = max(
                right - left_24h,
                0
            )

            idx = indices[i]

            df.at[
                idx,
                "TERMINAL_TX_COUNT_1H"
            ] = count_1h

            df.at[
                idx,
                "TERMINAL_TX_COUNT_24H"
            ] = count_24h

    # -----------------------------------------------------
    # Customer-terminal relationship
    #
    # Only strictly earlier timestamps count as history.
    # -----------------------------------------------------

    df["CUSTOMER_TERMINAL_TX_COUNT_BEFORE"] = 0

    for (
        customer_id,
        terminal_id
    ), group in df.groupby(
        ["CUSTOMER_ID", "TERMINAL_ID"],
        sort=False
    ):

        indices = group.index.to_numpy()

        times = (
            group["TX_DATETIME"]
            .astype("int64")
            .to_numpy()
        )

        for i, current_time in enumerate(times):

            right = np.searchsorted(
                times,
                current_time,
                side="left"
            )

            df.at[
                indices[i],
                "CUSTOMER_TERMINAL_TX_COUNT_BEFORE"
            ] = right

    df["CUSTOMER_TERMINAL_SEEN_BEFORE"] = (
        df["CUSTOMER_TERMINAL_TX_COUNT_BEFORE"] > 0
    ).astype(int)

    # -----------------------------------------------------
    # Customer amount anomaly
    # -----------------------------------------------------

    df["CUSTOMER_AMOUNT_ZSCORE"] = (
        (
            df["TX_AMOUNT"]
            - df["CUSTOMER_AVG_AMOUNT_BEFORE"]
        )
        /
        df["CUSTOMER_AMOUNT_STD_BEFORE"]
        .replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Clean numerical features
    # -----------------------------------------------------

    numerical_features = [
        "CUSTOMER_AVG_AMOUNT_BEFORE",
        "CUSTOMER_AMOUNT_STD_BEFORE",
        "CUSTOMER_AMOUNT_RATIO",
        "TERMINAL_AVG_AMOUNT_BEFORE",
        "TERMINAL_AMOUNT_RATIO",
        "CUSTOMER_TX_COUNT_1H",
        "CUSTOMER_TX_COUNT_24H",
        "CUSTOMER_AMOUNT_24H",
        "TERMINAL_TX_COUNT_1H",
        "TERMINAL_TX_COUNT_24H",
        "CUSTOMER_TERMINAL_TX_COUNT_BEFORE",
        "CUSTOMER_AMOUNT_ZSCORE",
    ]

    for column in numerical_features:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df):,} transactions."
    )

    print(
        "\nBuilding behavioural features..."
    )

    df = build_features(df)

    print(
        "Feature engineering complete."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved feature dataset to:"
        f"\n{OUTPUT_FILE}"
    )

    print("\nFinal shape:")
    print(df.shape)

    print("\nFinal columns:")

    for column in df.columns:
        print(f"  - {column}")


if __name__ == "__main__":
    main()