"""
What target leakage is worth, in points of PR-AUC.

The point-in-time discipline in `tribunal/features.py` is the most expensive design
decision in this project - it is the reason features have to be built by streaming
rather than with a groupby, and the reason label-derived features carry a delay. It
is worth showing what it buys, by building the two shortcuts it refuses to take and
measuring how much better they *appear* to be.

Three protocols, same model class, same data:

  1. **Honest** - temporal split, point-in-time features, labels delayed by the
     dispute window. This is what the deployed system does.

  2. **Global aggregates** - temporal split, but the card's average spend and the
     merchant's fraud rate are computed over the *entire* dataset, including
     transactions that had not happened yet. This is the single most common fraud
     modelling mistake, and it is invisible: the code looks like a normal groupby.

  3. **Random split + global aggregates** - the standard notebook setup. Rows are
     shuffled before splitting, so a card's April transactions train a model that is
     scored on that same card's February transactions.

Protocol 3 is the one that produces the 0.99 numbers people put in README files.
Only protocol 1 is achievable by a system that has to decide before it knows.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TRAIN_END = "2024-01-31"
VALID_END = "2024-03-31"

BASE = [
    "amt", "log_amt", "hour", "dow", "is_night", "age_years",
    "amt_z", "amt_over_card_mean", "card_txn_count",
    "cnt_1h", "cnt_24h", "cnt_7d", "secs_since_last", "category",
]
LEAKY = ["card_mean_amt_global", "merchant_fraud_rate_global", "card_fraud_rate_global"]

# The regime real fraud models live in: the transaction and the clock, but no
# trailing behavioural history. Published production card-fraud models report
# PR-AUC well below this benchmark's near-ceiling numbers.
BASE_WEAK = ["amt", "log_amt", "hour", "dow", "is_night", "age_years", "category"]


def add_global_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """The seductive shortcut: one groupby over everything.

    Nothing about this code looks wrong. It is wrong because every row's feature is
    computed from rows that, at decision time, were still in the future - including
    the labels of transactions that had not yet been disputed.
    """
    out = df.copy()
    out["card_mean_amt_global"] = out.groupby("cc_num")["amt_raw"].transform("mean")
    out["merchant_fraud_rate_global"] = out.groupby("merchant")["is_fraud"].transform("mean")
    out["card_fraud_rate_global"] = out.groupby("cc_num")["is_fraud"].transform("mean")
    return out


def fit_eval(tr, va, te, feats, seed=42):
    cat = [c for c in feats if c == "category"]
    mask = [c in cat for c in feats]

    def X(d):
        o = d[feats].copy()
        for c in cat:
            o[c] = o[c].astype("category")
        return o

    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=mask if cat else None,
        early_stopping=True, n_iter_no_change=25, random_state=seed,
    )
    m.fit(X(tr), tr["is_fraud"])
    cal = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic")
    cal.fit(X(va), va["is_fraud"])
    p = cal.predict_proba(X(te))[:, 1]
    y = te["is_fraud"].to_numpy()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_scored.parquet")
    ap.add_argument("--out", default="reports/leakage_ablation.json")
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    ts = pd.to_datetime(df["ts"])
    g = add_global_aggregates(df)

    tr, va, te = df[ts <= TRAIN_END], df[(ts > TRAIN_END) & (ts <= VALID_END)], df[ts > VALID_END]
    gtr, gva, gte = (g[ts <= TRAIN_END], g[(ts > TRAIN_END) & (ts <= VALID_END)], g[ts > VALID_END])
    rtr, rest = train_test_split(g, test_size=0.3, random_state=42, stratify=g["is_fraud"])
    rva, rte = train_test_split(rest, test_size=0.6, random_state=42, stratify=rest["is_fraud"])

    # Run the comparison twice. Once with the full feature set, where the honest
    # model is already close to the ceiling this benchmark allows and leakage has
    # little room to flatter it - and once with a deliberately weaker feature set,
    # which is the regime real production models actually operate in, and where the
    # size of the illusion becomes visible.
    all_results = {}
    for regime, base in (("full feature set", BASE), ("weak feature set", BASE_WEAK)):
        r = {
            "honest (temporal split, point-in-time features)": fit_eval(tr, va, te, base),
            "leaky aggregates (temporal split)": fit_eval(gtr, gva, gte, base + LEAKY),
            "random split + leaky aggregates": fit_eval(rtr, rva, rte, base + LEAKY),
        }
        all_results[regime] = r

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(all_results, fh, indent=2)

    for regime, r in all_results.items():
        print(f"\n=== {regime} ===")
        print(f"{'protocol':52s} {'ROC-AUC':>9s} {'PR-AUC':>9s}")
        print("-" * 72)
        for k, v in r.items():
            print(f"{k:52s} {v['roc_auc']:9.4f} {v['pr_auc']:9.4f}")
        h = r["honest (temporal split, point-in-time features)"]["pr_auc"]
        best = max(v["pr_auc"] for v in r.values())
        print(f"leakage inflates PR-AUC by up to {best - h:+.4f} "
              f"({100 * (best - h) / h:+.1f}% relative)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
