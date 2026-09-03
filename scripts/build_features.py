"""
Materialise the point-in-time feature matrix.

Streams the transaction table in timestamp order through FeatureEngine and writes
one row of features per transaction. Because the engine is streaming, this is the
only place features are ever produced - training, evaluation and the live API all
call the same code path, so there is no train/serve skew.

The bulk path writes each feature vector straight into preallocated arrays rather
than accumulating a million dicts, which keeps the whole 900k-row build inside a
few hundred megabytes.
"""

from __future__ import annotations

import argparse
import gc
import time
from operator import itemgetter

import numpy as np
import pandas as pd

from tribunal.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureConfig,
    FeatureEngine,
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the plain per-transaction fields the engine expects."""
    df = df.copy()
    # Convert to whole seconds via the dtype, not by dividing the integer
    # representation. pandas stores datetimes at microsecond resolution by default,
    # so `astype("int64") // 10**9` silently compresses an 18-month dataset into 47
    # seconds and quietly destroys every velocity and time-gap feature.
    df["unix_ts"] = df["ts"].astype("datetime64[s]").astype("int64")
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    dob = pd.to_datetime(df["dob"])
    df["age_years"] = (df["ts"] - dob).dt.days / 365.25
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--txns", default="data/transactions.parquet")
    p.add_argument("--out", default="data/features.parquet")
    p.add_argument("--label-delay-days", type=float, default=7.0)
    p.add_argument(
        "--no-label-features",
        action="store_true",
        help="disable every label-derived feature (strictest setting)",
    )
    a = p.parse_args()

    df = prepare(pd.read_parquet(a.txns))
    assert df["ts"].is_monotonic_increasing, "transactions must be time-ordered"
    n = len(df)

    cfg = FeatureConfig(
        label_delay_days=a.label_delay_days,
        disable_label_features=a.no_label_features,
    )
    eng = FeatureEngine(cfg)

    cols = [
        "unix_ts", "cc_num", "amt", "category", "merchant",
        "lat", "long", "merch_lat", "merch_long", "city_pop",
        "hour", "dow", "age_years", "is_fraud",
    ]
    records = df[cols].to_dict("records")

    num = np.empty((n, len(NUMERIC_FEATURES)), dtype=np.float32)
    cat = np.empty(n, dtype=object)
    numeric_names = NUMERIC_FEATURES
    # itemgetter pulls the whole row out of the dict in one C call, and assigning a
    # tuple into a numpy row converts it in C too. Doing this element-by-element in
    # Python costs ~26M scalar conversions and dominates the entire build.
    getvals = itemgetter(*numeric_names)

    t0 = time.time()
    gc.disable()  # nothing here is cyclic; refcounting is enough and this is ~2x
    try:
        for i, txn in enumerate(records):
            f = eng.compute(txn)
            eng.observe(txn)
            num[i] = getvals(f)
            cat[i] = f["category"]
            if i and i % 200_000 == 0:
                gc.collect()
                print(f"  {i:,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    finally:
        gc.enable()

    feats = pd.DataFrame(num, columns=numeric_names)
    for c in CATEGORICAL_FEATURES:
        feats[c] = pd.Categorical(cat)

    feats["is_fraud"] = df["is_fraud"].to_numpy()
    feats["ts"] = df["ts"].to_numpy()
    feats["amt_raw"] = df["amt"].to_numpy(dtype=np.float64)
    feats["cc_num"] = df["cc_num"].to_numpy()
    feats["merchant"] = df["merchant"].to_numpy()

    feats.to_parquet(a.out, index=False)
    print(
        f"wrote {a.out}  {n:,} rows x {len(numeric_names)} numeric + "
        f"{len(CATEGORICAL_FEATURES)} categorical in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
