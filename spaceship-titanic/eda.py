"""EDA for Spaceship Titanic.

Loads train.csv and test.csv from data/ and prints shapes, dtypes, the first
rows, the target distribution, and missingness per column. No modeling.

Download the data first, from the spaceship-titanic/ folder:
    kaggle competitions download -c spaceship-titanic -p data
    python -c "import zipfile; zipfile.ZipFile('data/spaceship-titanic.zip').extractall('data')"
"""

from pathlib import Path
import sys

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET = "Transported"


def load(name):
    path = DATA_DIR / name
    if not path.exists():
        sys.exit(f"missing {path}. Download the data into {DATA_DIR} first.")
    return pd.read_csv(path)


def describe(df, name):
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print("\ndtypes:")
    print(df.dtypes)
    print("\nhead:")
    print(df.head())
    print("\nmissing per column:")
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"n_missing": missing, "pct": pct}))


def main():
    train = load("train.csv")
    test = load("test.csv")
    describe(train, "train")
    describe(test, "test")
    if TARGET in train.columns:
        print(f"\n=== target: {TARGET} ===")
        counts = train[TARGET].value_counts(dropna=False)
        print(counts)
        print("\nproportion:")
        print((counts / len(train)).round(4))


if __name__ == "__main__":
    main()
