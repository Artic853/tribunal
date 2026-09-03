"""
The objective function.

A fraud system is not trying to maximise AUC. It is trying to maximise money kept,
subject to a fixed number of human analysts and a tolerance for annoying good
customers. Those are different problems and they rank policies differently.

Every constant below is an assumption, stated in one place so a reviewer can argue
with it. `eval/sensitivity.py` re-runs the whole evaluation across plausible ranges
so no conclusion rests on a single guess.

Currency note: the underlying benchmark is US-shaped and its `amt` column is in USD.
We treat one amount unit as one rupee throughout. The absolute rupee totals therefore
scale with that choice, but the *ranking* of policies is invariant to it, because
every term below is either proportional to amount or a fixed per-event cost that we
express in the same unit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict


class Decision(str, Enum):
    ALLOW = "allow"
    STEP_UP = "step_up"   # challenge with an additional factor (OTP / 3DS)
    REVIEW = "review"     # hold and queue for a human analyst
    BLOCK = "block"


@dataclass(frozen=True)
class CostModel:
    # --- fraud that gets through -------------------------------------------
    # The disputed amount is refunded to the cardholder, and the chargeback itself
    # carries a fixed operational fee (representment, network fee, ops time).
    chargeback_fixed_fee: float = 500.0

    # --- good customers we turn away ---------------------------------------
    # A false decline costs the lost margin on this sale plus lasting damage:
    # a materially large share of falsely-declined customers reduce or stop using
    # the instrument. We model it as a multiple of the transaction amount plus a
    # fixed relationship-damage term.
    false_decline_margin_rate: float = 0.15
    false_decline_fixed: float = 250.0

    # --- friction ----------------------------------------------------------
    # Probability a legitimate customer abandons when challenged with a step-up.
    step_up_abandon_rate: float = 0.06
    # Probability a step-up actually stops a fraudster (they lack the OTP/device).
    step_up_blocks_fraud_rate: float = 0.90
    # Fixed cost of issuing a step-up (SMS/vendor fee + measurable drop-off).
    step_up_fixed_cost: float = 15.0

    # --- manual review -----------------------------------------------------
    # Fully-loaded analyst cost of one case.
    review_cost: float = 120.0
    # Analyst accuracy: they are good, not perfect, and they are not free.
    review_catches_fraud_rate: float = 0.95
    review_clears_legit_rate: float = 0.97
    # A held case that an analyst eventually clears still delayed a good customer.
    review_delay_cost: float = 40.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def expected_cost(decision: Decision, amount: float, is_fraud: int, cm: CostModel) -> float:
    """Expected rupees lost by taking `decision` on a transaction of known outcome.

    Lower is better. ALLOW on a legitimate transaction is the zero point: it is the
    outcome the business wants, so it costs nothing. Everything else is measured as
    a loss relative to that.
    """
    a = float(amount)

    if decision is Decision.ALLOW:
        return a + cm.chargeback_fixed_fee if is_fraud else 0.0

    if decision is Decision.BLOCK:
        if is_fraud:
            return 0.0
        return cm.false_decline_margin_rate * a + cm.false_decline_fixed

    if decision is Decision.STEP_UP:
        if is_fraud:
            leaked = 1.0 - cm.step_up_blocks_fraud_rate
            return cm.step_up_fixed_cost + leaked * (a + cm.chargeback_fixed_fee)
        abandoned = cm.step_up_abandon_rate * (
            cm.false_decline_margin_rate * a + cm.false_decline_fixed
        )
        return cm.step_up_fixed_cost + abandoned

    if decision is Decision.REVIEW:
        base = cm.review_cost
        if is_fraud:
            missed = 1.0 - cm.review_catches_fraud_rate
            return base + missed * (a + cm.chargeback_fixed_fee)
        wrongly_declined = 1.0 - cm.review_clears_legit_rate
        return (
            base
            + cm.review_delay_cost
            + wrongly_declined * (cm.false_decline_margin_rate * a + cm.false_decline_fixed)
        )

    raise ValueError(f"unknown decision {decision!r}")


def do_nothing_cost(amount: float, is_fraud: int, cm: CostModel) -> float:
    """Cost of the null policy: approve everything."""
    return expected_cost(Decision.ALLOW, amount, is_fraud, cm)


def block_everything_cost(amount: float, is_fraud: int, cm: CostModel) -> float:
    """Cost of the paranoid policy: decline everything. Included because it is the
    other trivial bound, and a system that cannot beat *both* trivial bounds is not
    worth deploying."""
    return expected_cost(Decision.BLOCK, amount, is_fraud, cm)
