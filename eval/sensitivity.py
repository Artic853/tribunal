"""
How much do the conclusions depend on the cost model I made up?

Every constant in `tribunal/economics.py` is an assumption. None of them was
measured from a real payments business, because I do not have one. That is the
largest single weakness in this project's headline numbers, and the only honest
response is to find out how much the conclusions move when the assumptions do.

Two questions, both answered here:

1. **Is the ranking robust?** The README claims the *ordering* of policies survives
   the assumptions even though the rupee totals do not. That is a testable claim,
   so it is tested: each parameter is swept across a plausible range while the
   others hold, and at every point the expected-cost policy is compared against the
   best fixed score threshold - where the threshold policy is given the unfair
   advantage of being tuned on the test window itself.

2. **When does manual review start being worth it?** On the default parameters the
   policy never sends a case to a human. Sweeping the analyst cost shows exactly
   where that changes, which turns "the review queue is unused" from an
   embarrassment into a capacity-planning answer.

Output: reports/sensitivity.json and reports/sensitivity.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from eval.policies import expected_cost_vectorised
from tribunal.economics import CostModel

VALID_END = "2024-03-31"

# Index into ACTIONS: allow, step_up, review, block.
ALLOW, STEP_UP, REVIEW, BLOCK = 0, 1, 2, 3

# Plausible ranges. Defaults are marked in the report so the reader can see where
# the shipped configuration sits inside each sweep.
SWEEPS: Dict[str, List[float]] = {
    "chargeback_fixed_fee": [0, 100, 250, 500, 1000, 2000],
    "false_decline_margin_rate": [0.02, 0.05, 0.10, 0.15, 0.25, 0.40],
    "false_decline_fixed": [0, 50, 150, 250, 500, 1000],
    "step_up_abandon_rate": [0.01, 0.03, 0.06, 0.12, 0.20, 0.35],
    "step_up_blocks_fraud_rate": [0.50, 0.70, 0.80, 0.90, 0.95, 0.99],
    "review_cost": [10, 30, 60, 120, 250, 500],
    "review_catches_fraud_rate": [0.80, 0.90, 0.95, 0.98, 0.995],
}

THRESHOLD_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def cost_matrix(amt: np.ndarray, y: np.ndarray, cm: CostModel) -> np.ndarray:
    """Realised cost of each action on each transaction, given the true label.

    Shape (n, 4). Vectorised counterpart of `economics.expected_cost`; the
    equivalence is pinned by tests/test_vectorised_policy.py.
    """
    fd = cm.false_decline_margin_rate * amt + cm.false_decline_fixed
    loss = amt + cm.chargeback_fixed_fee
    f = y == 1
    g = ~f

    allow = np.where(f, loss, 0.0)
    block = np.where(f, 0.0, fd)
    step = cm.step_up_fixed_cost + np.where(
        f, (1 - cm.step_up_blocks_fraud_rate) * loss, cm.step_up_abandon_rate * fd
    )
    review = cm.review_cost + np.where(
        f,
        (1 - cm.review_catches_fraud_rate) * loss,
        cm.review_delay_cost + (1 - cm.review_clears_legit_rate) * fd,
    )
    return np.column_stack([allow, step, review, block])


def total_cost(codes: np.ndarray, costs: np.ndarray) -> float:
    return float(costs[np.arange(len(codes)), codes].sum())


def best_threshold(p, amt, y, costs) -> Tuple[float, float, np.ndarray]:
    """Best fixed block-threshold, tuned on the very window it is scored on.

    Giving the simplest baseline an oracle advantage is deliberate: if the
    expected-cost policy still wins, the result cannot be explained by tuning.
    """
    best = (None, np.inf, None)
    for thr in THRESHOLD_GRID:
        codes = np.where(p >= thr, BLOCK, ALLOW)
        c = total_cost(codes, costs)
        if c < best[1]:
            best = (thr, c, codes)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_scored.parquet")
    ap.add_argument("--out", default="reports/sensitivity.json")
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    te = df[pd.to_datetime(df["ts"]) > VALID_END]
    p = te["p_fraud"].to_numpy(dtype=np.float64)
    amt = te["amt_raw"].to_numpy(dtype=np.float64)
    y = te["is_fraud"].to_numpy()

    base = CostModel()
    defaults = base.as_dict()
    out: Dict[str, object] = {"defaults": defaults, "sweeps": {}}
    flips: List[str] = []

    for param, values in SWEEPS.items():
        rows = []
        for v in values:
            cm = replace(base, **{param: v})
            costs = cost_matrix(amt, y, cm)

            # Null policy: approve everything.
            status_quo = total_cost(np.zeros(len(p), dtype=int), costs)

            ec_codes = expected_cost_vectorised(p, amt, cm)
            ec_cost = total_cost(ec_codes, costs)

            thr, thr_cost, _ = best_threshold(p, amt, y, costs)

            ec_net = status_quo - ec_cost
            thr_net = status_quo - thr_cost
            wins = ec_net >= thr_net - 1e-6

            if not wins:
                flips.append(f"{param}={v}")

            rows.append({
                "value": v,
                "is_default": abs(v - defaults[param]) < 1e-12,
                "expected_cost_net_saved": round(ec_net, 2),
                "best_threshold_net_saved": round(thr_net, 2),
                "best_threshold_value": thr,
                "expected_cost_wins": bool(wins),
                "uplift_pct": round(100 * (ec_net - thr_net) / thr_net, 2) if thr_net > 0 else None,
                "reviews_wanted": int((ec_codes == REVIEW).sum()),
                "step_ups": int((ec_codes == STEP_UP).sum()),
                "blocks": int((ec_codes == BLOCK).sum()),
                "intervention_rate_pct": round(100 * float((ec_codes != ALLOW).mean()), 4),
            })
        out["sweeps"][param] = rows

        d = next(r for r in rows if r["is_default"]) if any(r["is_default"] for r in rows) else None
        print(f"\n{param}  (default {defaults[param]})")
        for r in rows:
            mark = " <- default" if r["is_default"] else ""
            print(f"   {r['value']:>8}  EC {r['expected_cost_net_saved']:>12,.0f}   "
                  f"thr {r['best_threshold_net_saved']:>12,.0f}   "
                  f"{'EC wins' if r['expected_cost_wins'] else 'THRESHOLD WINS'}   "
                  f"reviews {r['reviews_wanted']:>6,}{mark}")

    out["ranking_flips"] = flips
    out["ranking_robust"] = not flips

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    total = sum(len(v) for v in SWEEPS.values())
    print(f"\n{'=' * 70}")
    if flips:
        print(f"RANKING NOT ROBUST: the expected-cost policy loses at {len(flips)} of "
              f"{total} points: {', '.join(flips)}")
    else:
        print(f"Ranking robust: the expected-cost policy wins at all {total} "
              f"parameter settings tested, against an oracle-tuned threshold.")
    rc = out["sweeps"]["review_cost"]
    print("\nreview demand vs analyst cost: " + ", ".join(
        f"{r['value']}->{r['reviews_wanted']:,}" for r in rc))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
