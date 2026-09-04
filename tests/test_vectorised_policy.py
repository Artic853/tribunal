"""
The fast path must agree with the reference path.

`eval/policies.expected_cost_vectorised` is an optimisation, and an optimisation
that quietly disagrees with the thing it replaces is worse than no optimisation:
the sensitivity analysis would be measuring a policy nobody ships.

The reference implementation is the row-by-row one, because that is the code the
live agent actually runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.policies import ACTIONS, expected_cost_vectorised
from tribunal.economics import CostModel, Decision, expected_cost
from tribunal.policy import decide, expected_costs


def reference(p: float, amt: float, cm: CostModel) -> Decision:
    return decide(p_fraud=p, amount=amt, evidence=[], cm=cm, budget=None).unconstrained


@pytest.mark.parametrize("cm", [
    CostModel(),
    CostModel(review_cost=20.0),
    CostModel(review_cost=5000.0),
    CostModel(false_decline_margin_rate=0.5, false_decline_fixed=1000.0),
    CostModel(chargeback_fixed_fee=0.0),
    CostModel(step_up_blocks_fraud_rate=0.5, step_up_abandon_rate=0.3),
])
def test_matches_reference_across_cost_models(cm):
    rng = np.random.default_rng(0)
    # Probabilities spanning the full range, amounts spanning four orders of
    # magnitude - the disagreements, if any, live at the decision boundaries.
    p = np.concatenate([
        rng.random(400),
        rng.random(400) ** 8,            # crowd the near-zero region
        1 - rng.random(200) ** 8,        # and the near-one region
    ])
    amt = 10 ** (rng.random(len(p)) * 4)

    codes = expected_cost_vectorised(p, amt, cm)
    fast = [ACTIONS[c] for c in codes]
    slow = [reference(float(pi), float(ai), cm) for pi, ai in zip(p, amt)]

    mismatches = [
        (pi, ai, f, s) for pi, ai, f, s in zip(p, amt, fast, slow) if f is not s
    ]
    assert not mismatches, (
        f"{len(mismatches)} of {len(p)} disagree, first: "
        f"p={mismatches[0][0]:.6f} amt={mismatches[0][1]:.2f} "
        f"fast={mismatches[0][2].value} slow={mismatches[0][3].value}"
    )


def test_expected_costs_match_termwise():
    """Not just the argmin - the costs themselves must agree, or the review-value
    calculation the budget controller uses would be wrong."""
    cm = CostModel()
    for p in (0.0, 1e-6, 0.01, 0.3, 0.5, 0.87, 1.0):
        for amt in (50.0, 999.0, 25_000.0):
            ref = expected_costs(p, amt, cm)
            for d in ACTIONS:
                manual = (
                    p * expected_cost(d, amt, 1, cm)
                    + (1 - p) * expected_cost(d, amt, 0, cm)
                )
                assert ref[d] == pytest.approx(manual, rel=1e-12)
