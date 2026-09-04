"""
Policies under test.

Each policy is a function from the precomputed point-in-time feature frame to one
Decision per transaction. Running them over the *same* feature matrix that the live
agent computes is what makes the comparison fair: no policy gets better information
than another, and none of them can see the future, because the feature engine could
not.

The two trivial policies at the top are here on purpose. A fraud system that cannot
beat "approve everything" is losing money; one that cannot beat "decline everything"
has not understood the cost of false declines. Both bounds get reported every time.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from tribunal.economics import CostModel, Decision
from tribunal.policy import ACTIONS, ReviewBudget, decide, expected_costs
from tribunal.tools.base import Evidence
from tribunal.tools.heuristics import registry

DAY_S = 86400.0

PolicyFn = Callable[[pd.DataFrame, CostModel], List[Decision]]


# --------------------------------------------------------------- trivial bounds

def allow_all(df: pd.DataFrame, cm: CostModel) -> List[Decision]:
    return [Decision.ALLOW] * len(df)


def block_all(df: pd.DataFrame, cm: CostModel) -> List[Decision]:
    return [Decision.BLOCK] * len(df)


# ------------------------------------------------------------------ hand rules

def rules_only(df: pd.DataFrame, cm: CostModel) -> List[Decision]:
    """A plausible hand-written rule set - what a risk team has before any model.

    These are not strawmen: each rule encodes a real pattern, and the thresholds
    were chosen on the validation window (see scripts/calibrate_rules.py), not on
    the test window they are scored against.
    """
    amt = df["amt"].to_numpy()
    night = df["is_night"].to_numpy() == 1
    new_m = df["is_new_merchant"].to_numpy() == 1
    c1 = np.nan_to_num(df["cnt_1h"].to_numpy())
    z = np.nan_to_num(df["amt_z"].to_numpy())
    speed = np.nan_to_num(df["implied_speed_kmh"].to_numpy())

    block = (
        ((amt > 700) & night & new_m)
        | (c1 >= 6)
        | (speed > 1200)
        | (z > 8)
    )
    step = (~block) & (
        ((amt > 400) & night)
        | (z > 4)
        | (c1 >= 4)
    )

    # Built as a Python list rather than an object array: Decision subclasses str,
    # so numpy quietly downcasts enum members to bare strings on assignment.
    return [
        Decision.BLOCK if b else (Decision.STEP_UP if s else Decision.ALLOW)
        for b, s in zip(block, step)
    ]


# ------------------------------------------------------------- model threshold

def model_threshold_factory(threshold: float) -> PolicyFn:
    """Block above a fixed score, approve below. The usual student answer, and the
    thing an expected-cost policy has to beat to justify its complexity."""

    def fn(df: pd.DataFrame, cm: CostModel) -> List[Decision]:
        p = df["p_fraud"].to_numpy()
        return [Decision.BLOCK if x >= threshold else Decision.ALLOW for x in p]

    fn.__name__ = f"model_threshold_{threshold:g}"
    return fn


# --------------------------------------------------------- expected-cost family

def expected_cost_policy(
    df: pd.DataFrame,
    cm: CostModel,
    budget: Optional[ReviewBudget] = None,
    use_overrides: bool = True,
    use_evidence: bool = True,
) -> List[Decision]:
    """Lowest expected cost per transaction, optionally under review capacity and
    with deterministic escalation rules."""
    p = df["p_fraud"].to_numpy()
    amt = df["amt_raw"].to_numpy()
    ts = pd.to_datetime(df["ts"]).astype("datetime64[s]").astype("int64").to_numpy()
    epoch = ts[0]

    records = df.to_dict("records") if use_evidence else None
    empty: List[Evidence] = []

    out: List[Decision] = []
    for i in range(len(df)):
        ev = registry.run_all(records[i]) if (use_evidence and use_overrides) else empty
        res = decide(
            p_fraud=float(p[i]),
            amount=float(amt[i]),
            evidence=ev if use_overrides else empty,
            cm=cm,
            budget=budget,
            day_index=int((ts[i] - epoch) // DAY_S),
        )
        out.append(res.decision)
    return out


def make_tribunal(capacity_per_day: int, use_overrides: bool = True) -> PolicyFn:
    def fn(df: pd.DataFrame, cm: CostModel) -> List[Decision]:
        budget = ReviewBudget(capacity_per_day=capacity_per_day) if capacity_per_day else None
        return expected_cost_policy(
            df, cm,
            budget=budget,
            use_overrides=use_overrides,
            use_evidence=use_overrides,
        )

    fn.__name__ = f"tribunal_cap{capacity_per_day}" + ("" if use_overrides else "_no_overrides")
    return fn


def expected_cost_vectorised(
    p: np.ndarray, amt: np.ndarray, cm: CostModel
) -> np.ndarray:
    """Expected-cost minimisation over the four actions, in closed form.

    Every term in `economics.expected_cost` is linear in the amount, so the whole
    policy is four columns of arithmetic and an argmin. The row-by-row version in
    `expected_cost_policy` stays the reference implementation - it is the one the
    live agent shares - and `tests/test_vectorised_policy.py` asserts the two agree
    exactly on a sample.

    This exists because the sensitivity analysis re-runs the policy a few hundred
    times over 160,000 rows, and at ~11 s per pass the loop would turn a two-minute
    experiment into an hour.

    Returns an integer array indexing ACTIONS. Review capacity is *not* modelled
    here: this is the unlimited-review policy, which is the right object for asking
    how the cost parameters change what the policy *wants* to do.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    a = np.asarray(amt, dtype=np.float64)
    q = 1.0 - p

    fd = cm.false_decline_margin_rate * a + cm.false_decline_fixed   # cost of a false decline
    loss = a + cm.chargeback_fixed_fee                               # cost of letting fraud through

    allow = p * loss
    block = q * fd
    step = (
        cm.step_up_fixed_cost
        + p * (1.0 - cm.step_up_blocks_fraud_rate) * loss
        + q * cm.step_up_abandon_rate * fd
    )
    review = (
        cm.review_cost
        + p * (1.0 - cm.review_catches_fraud_rate) * loss
        + q * (cm.review_delay_cost + (1.0 - cm.review_clears_legit_rate) * fd)
    )

    # Column order must match ACTIONS.
    return np.argmin(np.column_stack([allow, step, review, block]), axis=1)


def decisions_from_codes(codes: np.ndarray) -> List[Decision]:
    return [ACTIONS[c] for c in codes]


def expected_cost_unlimited(df: pd.DataFrame, cm: CostModel) -> List[Decision]:
    """Expected-cost minimisation with no capacity constraint and no rules. This is
    the theoretical ceiling of the cost model, and it is not achievable - it sends
    far more cases to review than any team can staff. Reported so the gap between
    the ideal and the deliverable policy is visible."""
    return expected_cost_policy(df, cm, budget=None, use_overrides=False, use_evidence=False)
