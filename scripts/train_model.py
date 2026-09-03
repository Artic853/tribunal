"""
Train and calibrate the risk model.

Three choices here are deliberate and worth defending:

1. **Temporal split, not random split.** Fraud is non-stationary and cards repeat.
   A random split lets the model see a card's April behaviour while scoring its
   February behaviour. We train on the past and test on the future, once.

2. **No class reweighting.** The policy consumes `p(fraud)` as an actual
   probability - it multiplies it by rupees to get an expected cost. Reweighting a
   0.7%-positive problem to balance produces a model whose "0.8" means nothing.
   We keep the true prior and then calibrate.

3. **Isotonic calibration on a held-out window.** Gradient boosting is a good
   ranker and a mediocre probability estimator. Calibration is fitted on the
   validation window - never on train, never on test.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from tribunal.features import CATEGORICAL_FEATURES, MODEL_FEATURES

TRAIN_END = "2024-01-31"
VALID_END = "2024-03-31"


def temporal_split(df: pd.DataFrame):
    ts = pd.to_datetime(df["ts"])
    tr = df[ts <= TRAIN_END]
    va = df[(ts > TRAIN_END) & (ts <= VALID_END)]
    te = df[ts > VALID_END]
    return tr, va, te


def to_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[MODEL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features.parquet")
    p.add_argument("--out", default="artifacts/model.pkl")
    p.add_argument("--report", default="reports/model_metrics.json")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    tr, va, te = temporal_split(df)
    print(f"train {len(tr):,} ({tr.is_fraud.mean():.4%})  "
          f"valid {len(va):,} ({va.is_fraud.mean():.4%})  "
          f"test {len(te):,} ({te.is_fraud.mean():.4%})")

    cat_mask = [c in CATEGORICAL_FEATURES for c in MODEL_FEATURES]

    base = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        categorical_features=cat_mask,
        random_state=a.seed,
    )

    t0 = time.time()
    base.fit(to_X(tr), tr["is_fraud"])
    print(f"fitted base model in {time.time() - t0:.0f}s "
          f"({base.n_iter_} boosting rounds)")

    # Calibrate on the validation window only. FrozenEstimator keeps the fitted base
    # model exactly as it is, so only the isotonic mapping is learned here - the
    # model never sees the calibration window as training data.
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    cal.fit(to_X(va), va["is_fraud"])

    metrics = {}
    for name, part in (("train", tr), ("valid", va), ("test", te)):
        y = part["is_fraud"].to_numpy()
        p_raw = base.predict_proba(to_X(part))[:, 1]
        p_cal = cal.predict_proba(to_X(part))[:, 1]
        metrics[name] = {
            "n": int(len(part)),
            "fraud_rate": float(y.mean()),
            "roc_auc_raw": float(roc_auc_score(y, p_raw)),
            "roc_auc_calibrated": float(roc_auc_score(y, p_cal)),
            "pr_auc_raw": float(average_precision_score(y, p_raw)),
            "pr_auc_calibrated": float(average_precision_score(y, p_cal)),
            "brier_raw": float(brier_score_loss(y, p_raw)),
            "brier_calibrated": float(brier_score_loss(y, p_cal)),
        }
        print(f"{name:6s} ROC-AUC {metrics[name]['roc_auc_calibrated']:.4f}  "
              f"PR-AUC {metrics[name]['pr_auc_calibrated']:.4f}  "
              f"Brier {metrics[name]['brier_raw']:.5f} -> "
              f"{metrics[name]['brier_calibrated']:.5f}")

    # Score quantiles from the training window, stored with the model so that a
    # memo can say "top 0.1% of scored traffic" rather than only "0.83". An analyst
    # cannot act on a bare probability without knowing what normal looks like, and
    # computing this at serve time would mean the phrasing drifted with traffic.
    p_train = cal.predict_proba(to_X(tr))[:, 1]
    reference = {
        f"q{q}": float(np.quantile(p_train, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(
            {"model": cal, "base": base, "features": MODEL_FEATURES,
             "categorical": CATEGORICAL_FEATURES, "train_end": TRAIN_END,
             "valid_end": VALID_END, "seed": a.seed, "reference": reference},
            fh,
        )

    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Score the whole table once so downstream evaluation is cheap and consistent.
    df["p_fraud"] = cal.predict_proba(to_X(df))[:, 1]
    df.to_parquet(a.features.replace(".parquet", "_scored.parquet"), index=False)

    print(f"wrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
