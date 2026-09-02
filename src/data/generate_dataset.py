from pathlib import Path
import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Number of days to use for our first development dataset
N_DAYS = 30


def load_transactions(n_days: int = N_DAYS) -> pd.DataFrame:
    """Load the first n daily transaction files."""

    files = sorted(RAW_DATA_DIR.glob("*.pkl"))[:n_days]

    if not files:
        raise FileNotFoundError(
            f"No .pkl files found in {RAW_DATA_DIR}"
        )

    print(f"Found {len(files)} transaction files.")

    frames = []

    for file in files:
        print(f"Loading {file.name}...")
        df = pd.read_pickle(file)
        frames.append(df)

    transactions = pd.concat(frames, ignore_index=True)

    return transactions


def main():
    transactions = load_transactions()

    # Sort chronologically
    transactions = transactions.sort_values("TX_DATETIME").reset_index(drop=True)

    # Basic information
    print("\n--- Dataset summary ---")
    print(f"Rows: {len(transactions):,}")
    print(f"Columns: {len(transactions.columns)}")

    print("\nColumns:")
    print(list(transactions.columns))

    print("\nFraud distribution:")
    print(transactions["TX_FRAUD"].value_counts())

    fraud_rate = transactions["TX_FRAUD"].mean() * 100
    print(f"\nFraud rate: {fraud_rate:.3f}%")

    print("\nFirst five transactions:")
    print(transactions.head())

    # Save processed dataset
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DATA_DIR / "transactions_30d.csv"
    transactions.to_csv(output_file, index=False)

    print(f"\nSaved dataset to: {output_file}")


if __name__ == "__main__":
    main()