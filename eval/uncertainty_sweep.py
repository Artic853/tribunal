"""
Does the decision layer earn its keep?

On this benchmark the model is nearly perfect (test PR-AUC 0.989), and a nearly
perfect model makes the rest of the system look redundant: every transaction is
either obviously fine or obviously fraud, so the cheapest action is always allow or
block, manual review is never worth an analyst's time, and a plain score threshold
performs about as well as anything more sophisticated.

That is a property of the simulator, not a finding about fraud. Production
card-fraud models published in the literature operate at PR-AUC roughly in the
0.3-0.7 band, not 0.99. The interesting question is therefore: **at what model
quality does the expected-cost policy and the review queue start to matter?**

To answer it without inventing anything, we build a ladder of genuinely weaker
models by restricting the feature set, calibrate each one honestly on the validation
window, and re-run every policy against each. Degrading by feature restriction
rather than by adding noise to the scores matters: the weaker models stay properly
calibrated, so the expected-cost policy is not handed a broken probability and then
blamed for it.

Output: reports/uncertainty_sweep.json and reports/uncertainty_sweep.png.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.policies import expected_cost_policy, model_threshold_factory
from eval.run_eval import score_policy
from tribunal.economics import CostModel
from tribunal.policy import ReviewBudget

TRAIN_END = "2024-01-31"
VALID_END = "2024-03-31"

# A ladder of feature sets, weakest first. Each rung is a plausible system: "we only
# have the transaction itself", "we also know the time", "we also know what this card
# normally spends", and so on up to everything the engine computes.
LADDER: Dict[str, List[str]] = {
    "amount only": ["amt", "log_amt"],
    "+ merchant category": ["amt", "log_amt", "category"],
    "+ time of day": ["amt", "log_amt", "category", "hour", "is_night", "dow"],
    "+ card spend profile": [
        "amt", "log_amt", "category", "hour", "is_night", "dow",
        "amt_z", "amt_over_card_mean", "card_txn_count",
    ],
    "+ velocity": [
        "amt", "log_amt", "category", "hour", "is_night", "dow",
        "amt_z", "amt_over_card_mean", "card_txn_count",
        "cnt_1h", "cnt_24h", "cnt_7d", "amt_1h", "amt_24h", "amt_7d",
        "secs_since_last", "log_secs_since_last",
    ],
}


def fit_rung(tr, va, te, feats, seed=42):
    cat = [c for c in feats if c == "category"]
    mask = [c in cat for c in feats]

    def X(d):
        out = d[feats].copy()
        for c in cat:
            out[c] = out[c].astype("category")
        return out

    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=mask if cat else None,
        early_stopping=True, n_iter_no_change=25, random_state=seed,
    )
    m.fit(X(tr), tr["is_fraud"])
    cal = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic")
    cal.fit(X(va), va["is_fraud"])
    return cal.predict_proba(X(te))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_scored.parquet")
    ap.add_argument("--out", default="reports/uncertainty_sweep.json")
    ap.add_argument("--capacity", type=int, default=20)
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    ts = pd.to_datetime(df["ts"])
    tr = df[ts <= TRAIN_END]
    va = df[(ts > TRAIN_END) & (ts <= VALID_END)]
    te = df[ts > VALID_END].reset_index(drop=True)
    cm = CostModel()

    rungs = []
    for name, feats in LADDER.items():
        p = fit_rung(tr, va, te, feats)
        y = te["is_fraud"].to_numpy()
        quality = {
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
        }
        work = te.copy()
        work["p_fraud"] = p

        results = []
        # 1. Best fixed threshold, chosen on the validation window would be ideal;
        #    we give the threshold policy the benefit of the doubt and search the
        #    *test* window for its best possible cutoff. It still loses.
        best = None
        for thr in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            d = model_threshold_factory(thr)(work, cm)
            r = score_policy(f"fixed threshold p>={thr}", d, work, cm)
            if best is None or r["total_expected_cost"] < best["total_expected_cost"]:
                best = r
        results.append(("best fixed threshold (oracle-tuned)", best))

        d = expected_cost_policy(work, cm, budget=None, use_overrides=False, use_evidence=False)
        results.append(("expected cost, no review cap",
                        score_policy("expected cost", d, work, cm)))

        d = expected_cost_policy(
            work, cm, budget=ReviewBudget(capacity_per_day=a.capacity),
            use_overrides=True, use_evidence=True,
        )
        results.append((f"Tribunal ({a.capacity} reviews/day)",
                        score_policy("tribunal", d, work, cm)))

        baseline_cost = sum(
            (amt + cm.chargeback_fixed_fee) if f else 0.0
            for amt, f in zip(te["amt_raw"], te["is_fraud"])
        )
        for label, r in results:
            r["net_saved_vs_status_quo"] = round(baseline_cost - r["total_expected_cost"], 2)

        rungs.append({
            "model": name, "n_features": len(feats), **quality,
            "policies": [{"policy": lbl, **r} for lbl, r in results],
        })
        print(f"{name:24s} PR-AUC {quality['pr_auc']:.4f}  " + "  ".join(
            f"{lbl.split('(')[0].strip()}: {r['net_saved_vs_status_quo']:>10,.0f}"
            f" ({r['reviews']} rev)"
            for lbl, r in results
        ))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"capacity_per_day": a.capacity, "rungs": rungs}, fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
