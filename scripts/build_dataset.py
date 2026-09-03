"""
Consolidate raw Sparkov output into a single, time-ordered transaction table.

The raw data is produced by the open-source Sparkov simulator (see README for the
exact command and seed). This script only reshapes it: no filtering, no sampling,
no label manipulation.

Output: data/transactions.parquet, sorted strictly by event time.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

KEEP = [
    "trans_num",
    "ts",
    "cc_num",
    "amt",
    "category",
    "merchant",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
    "dob",
    "gender",
    "state",
    "job",
    "profile",
    "is_fraud",
]


def build(raw_dir: str, out_path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no csv files under {raw_dir!r} - run scripts/generate_data.sh first")

    frames = []
    for f in files:
        d = pd.read_csv(f, sep="|")
        if len(d):
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # The generator emits one customer-header row per customer with all transaction
    # columns null (customers whose chunk produced no transactions). Drop them and
    # say so out loud rather than silently.
    n_before = len(df)
    df = df[df["trans_num"].notna()].copy()
    dropped = n_before - len(df)
    if dropped:
        print(f"dropped {dropped:,} generator header rows with no transaction payload")

    # The generator prefixes every merchant name with "fraud_" - all 693 of them,
    # regardless of whether any of their transactions are fraudulent (659 merchants
    # carry a mix of both labels, none is wholly fraud). It is a naming quirk, not a
    # leak, but left in place it makes every case memo read as though the merchant
    # were already known to be criminal, so it is stripped here.
    df["merchant"] = df["merchant"].str.replace(r"^fraud_", "", regex=True)

    df["ts"] = pd.to_datetime(df["trans_date"] + " " + df["trans_time"])
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["cc_num"] = df["cc_num"].astype("int64")

    df = df[KEEP].sort_values("ts", kind="mergesort").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="../raw")
    p.add_argument("--out", default="data/transactions.parquet")
    a = p.parse_args()

    df = build(a.raw, a.out)
    n_fraud = int(df.is_fraud.sum())
    print(f"rows                {len(df):,}")
    print(f"fraud               {n_fraud:,} ({n_fraud / len(df):.4%})")
    print(f"cards               {df.cc_num.nunique():,}")
    print(f"merchants           {df.merchant.nunique():,}")
    print(f"window              {df.ts.min()} -> {df.ts.max()}")
    print(f"fraud value         {df.loc[df.is_fraud == 1, 'amt'].sum():,.0f}")
    print(f"legit value         {df.loc[df.is_fraud == 0, 'amt'].sum():,.0f}")
    print(f"wrote               {a.out}")


if __name__ == "__main__":
    main()
