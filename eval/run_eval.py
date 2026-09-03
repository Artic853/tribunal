"""
The evaluation.

Everything is scored on the held-out future window only, with a model that never
saw it and features that could not see it. Results are reported in rupees first,
because that is the quantity anyone deciding whether to deploy this actually cares
about, and in classification metrics second.

Reported for every policy:

  * net rupees saved against approving everything (the status quo),
  * fraud value stopped and fraud value missed,
  * good customers turned away, in count and in rupees of friction inflicted,
  * manual reviews consumed, against the capacity that was available,
  * precision and recall of the intervention.

The false-decline column is not an afterthought. A policy that blocks 3% of good
traffic to catch a little more fraud is a worse business than the fraud it prevents,
and the only way to see that is to price it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from eval.policies import (
    allow_all,
    block_all,
    expected_cost_unlimited,
    make_tribunal,
    model_threshold_factory,
    rules_only,
)
from tribunal.economics import CostModel, Decision, expected_cost

VALID_END = "2024-03-31"


def score_policy(
    name: str, decisions: List[Decision], df: pd.DataFrame, cm: CostModel
) -> Dict[str, object]:
    y = df["is_fraud"].to_numpy()
    amt = df["amt_raw"].to_numpy()
    d = np.array([x.value for x in decisions])

    total_cost = sum(
        expected_cost(dec, float(a), int(yy), cm)
        for dec, a, yy in zip(decisions, amt, y)
    )

    intervened = d != Decision.ALLOW.value
    blocked = d == Decision.BLOCK.value
    stepped = d == Decision.STEP_UP.value
    reviewed = d == Decision.REVIEW.value

    fraud = y == 1
    legit = ~fraud

    tp = int((intervened & fraud).sum())
    fp = int((intervened & legit).sum())
    fn = int((~intervened & fraud).sum())

    fraud_value_total = float(amt[fraud].sum())
    fraud_value_stopped = float(amt[intervened & fraud].sum())
    fraud_value_missed = float(amt[~intervened & fraud].sum())

    # Friction inflicted on people who did nothing wrong.
    false_decline_cost = sum(
        expected_cost(dec, float(a), 0, cm)
        for dec, a, yy in zip(decisions, amt, y)
        if yy == 0 and dec is not Decision.ALLOW
    )

    return {
        "policy": name,
        "total_expected_cost": round(total_cost, 2),
        "fraud_value_total": round(fraud_value_total, 2),
        "fraud_value_stopped": round(fraud_value_stopped, 2),
        "fraud_value_missed": round(fraud_value_missed, 2),
        "fraud_value_stopped_pct": round(100 * fraud_value_stopped / max(fraud_value_total, 1), 2),
        "interventions": int(intervened.sum()),
        "intervention_rate_pct": round(100 * intervened.mean(), 4),
        "blocks": int(blocked.sum()),
        "step_ups": int(stepped.sum()),
        "reviews": int(reviewed.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "good_customers_touched": fp,
        "good_customers_touched_pct": round(100 * fp / max(int(legit.sum()), 1), 4),
        "false_decline_cost": round(false_decline_cost, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features_scored.parquet")
    p.add_argument("--out", default="reports/eval.json")
    p.add_argument("--capacity", type=int, default=20, help="manual reviews per day")
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    test = df[pd.to_datetime(df["ts"]) > VALID_END].reset_index(drop=True)
    cm = CostModel()

    days = (pd.to_datetime(test["ts"]).max() - pd.to_datetime(test["ts"]).min()).days + 1
    print(f"test window: {len(test):,} transactions over {days} days")
    print(f"             {int(test.is_fraud.sum()):,} fraud "
          f"({test.is_fraud.mean():.4%}), "
          f"{test.loc[test.is_fraud == 1, 'amt_raw'].sum():,.0f} at risk\n")

    policies = [
        ("approve everything", allow_all),
        ("decline everything", block_all),
        ("hand-written rules", rules_only),
        ("model, block at p>=0.5", model_threshold_factory(0.5)),
        ("model, block at p>=0.9", model_threshold_factory(0.9)),
        ("expected cost, unlimited review", expected_cost_unlimited),
        (f"Tribunal ({a.capacity} reviews/day)", make_tribunal(a.capacity)),
        (f"Tribunal, no escalation rules", make_tribunal(a.capacity, use_overrides=False)),
    ]

    rows = []
    for name, fn in policies:
        decisions = fn(test, cm)
        rows.append(score_policy(name, decisions, test, cm))
        print(f"  scored: {name}")

    baseline = next(r for r in rows if r["policy"] == "approve everything")
    for r in rows:
        r["net_saved_vs_status_quo"] = round(
            baseline["total_expected_cost"] - r["total_expected_cost"], 2
        )

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(
            {"window_days": days, "n_transactions": len(test),
             "cost_model": cm.as_dict(), "results": rows},
            fh, indent=2,
        )

    # Console table
    print()
    hdr = f"{'policy':34s} {'net saved':>13s} {'fraud stopped':>14s} {'good hit':>9s} {'reviews':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['policy']:34s} {r['net_saved_vs_status_quo']:13,.0f} "
              f"{r['fraud_value_stopped_pct']:13.1f}% "
              f"{r['good_customers_touched']:9,d} {r['reviews']:8,d}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
