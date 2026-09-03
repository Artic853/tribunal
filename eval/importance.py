"""
What is the model actually using?

A near-perfect score is a reason for suspicion, not celebration. Permutation
importance on the held-out window answers the question a panel will ask first: is
this model finding fraud, or has it found a shortcut?

Permutation, rather than the model's internal split gains: gain-based importance
rewards features that were merely *convenient* to split on, and is computed on
training data. Permutation measures what the model loses on unseen data when one
feature is replaced by noise, which is the thing we actually care about.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

VALID_END = "2024-03-31"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_scored.parquet")
    ap.add_argument("--model", default="artifacts/model.pkl")
    ap.add_argument("--out", default="reports/importance.json")
    ap.add_argument("--sample", type=int, default=60000)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    with open(a.model, "rb") as fh:
        bundle = pickle.load(fh)
    model, feats, cats = bundle["model"], bundle["features"], bundle["categorical"]

    df = pd.read_parquet(a.features)
    te = df[pd.to_datetime(df["ts"]) > VALID_END]

    # Keep every fraud case and subsample the negatives: permutation importance on
    # 160k rows x 30 features x 3 repeats is slow, and dropping positives would make
    # the average-precision estimate noisy.
    pos = te[te.is_fraud == 1]
    neg = te[te.is_fraud == 0].sample(
        min(a.sample - len(pos), int((te.is_fraud == 0).sum())), random_state=42
    )
    s = pd.concat([pos, neg]).sort_values("ts")

    X = s[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    y = s["is_fraud"].to_numpy()

    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    print(f"baseline PR-AUC on {len(s):,} rows ({int(y.sum())} fraud): {base:.4f}")

    r = permutation_importance(
        model, X, y, scoring="average_precision",
        n_repeats=a.repeats, random_state=42, n_jobs=1,
    )

    rows = sorted(
        (
            {"feature": f, "drop": float(m), "std": float(sd)}
            for f, m, sd in zip(feats, r.importances_mean, r.importances_std)
        ),
        key=lambda d: -d["drop"],
    )

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"baseline_pr_auc": base, "importances": rows}, fh, indent=2)

    print(f"\n{'feature':26s} {'PR-AUC drop when shuffled':>26s}")
    print("-" * 54)
    for d in rows[:14]:
        bar = "#" * int(60 * d["drop"] / max(rows[0]["drop"], 1e-9))
        print(f"{d['feature']:26s} {d['drop']:>10.4f}  {bar}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
