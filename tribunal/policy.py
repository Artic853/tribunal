"""
Decision policy.

Given a calibrated probability of fraud and the amount at stake, the optimal action
is the one with the lowest expected cost. That is not a threshold on the score - it
depends on the amount, because declining a 200-rupee transaction and declining a
90,000-rupee transaction are not the same bet.

Two things complicate the textbook answer, and both are the point of this module:

1. **Manual review is a scarce resource.** There are a fixed number of analysts.
   The expected-cost calculation will happily send 4% of traffic to review; the
   queue holds 0.1%. So review admission runs through a budget controller that
   spends the day's capacity on the cases where review is worth the most, not the
   cases that merely score highest.

2. **The model is not the only source of truth.** Deterministic checks catch things
   a model trained on last quarter's fraud has never seen. They are allowed to
   escalate a decision, never to silently relax one, and every escalation is
   recorded so its value can be measured rather than assumed
   (`eval/override_analysis.py` reports whether the overrides actually helped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .economics import CostModel, Decision, expected_cost
from .tools.base import Evidence

# Actions the policy may choose from, cheapest-friction first.
ACTIONS: Tuple[Decision, ...] = (
    Decision.ALLOW,
    Decision.STEP_UP,
    Decision.REVIEW,
    Decision.BLOCK,
)


def expected_costs(p_fraud: float, amount: float, cm: CostModel) -> Dict[Decision, float]:
    """Expected rupee cost of each action under the model's probability."""
    p = min(max(float(p_fraud), 0.0), 1.0)
    return {
        d: p * expected_cost(d, amount, 1, cm) + (1.0 - p) * expected_cost(d, amount, 0, cm)
        for d in ACTIONS
    }


@dataclass
class ReviewBudget:
    """Daily manual-review capacity, enforced by an adaptive admission threshold.

    A fixed score cutoff cannot hold a queue at a fixed size: volume and fraud rates
    move, and the queue either starves or overflows. Instead we admit a case when the
    *value of reviewing it* - how many rupees review saves over the best action we
    could take without an analyst - clears a bar, and we move that bar with a simple
    proportional controller so the day's admissions land near capacity.

    This is intentionally the simplest controller that works. Its behaviour is
    visible in reports/review_queue.png; the queue tracks capacity within a few
    percent after the first week and does not oscillate.
    """

    capacity_per_day: int = 200
    threshold: float = 50.0      # rupees of expected saving required to admit
    min_threshold: float = 1.0
    max_threshold: float = 100_000.0
    gain: float = 0.25           # controller aggressiveness

    _day: Optional[int] = field(default=None, repr=False)
    _used_today: int = field(default=0, repr=False)
    _demand_today: int = field(default=0, repr=False)
    history: List[Tuple[int, int, int, float]] = field(default_factory=list, repr=False)

    def _roll(self, day_index: int) -> None:
        if self._day is None:
            self._day = day_index
            return
        while self._day < day_index:
            self.history.append(
                (self._day, self._used_today, self._demand_today, self.threshold)
            )
            # Proportional update on *demand*, not on admissions.
            #
            # Admissions are clamped at capacity, so a queue that is massively
            # oversubscribed looks identical to one that is exactly full - the
            # controller sees error == 0 and never raises the bar. Steering on the
            # number of cases that cleared the threshold gives it the signal it
            # needs. (Found by test_controller_raises_the_bar_when_over_capacity.)
            error = (self._demand_today - self.capacity_per_day) / max(self.capacity_per_day, 1)
            self.threshold *= 1.0 + self.gain * error
            self.threshold = min(max(self.threshold, self.min_threshold), self.max_threshold)
            self._day += 1
            self._used_today = 0
            self._demand_today = 0

    def admit(self, day_index: int, review_value: float) -> bool:
        """Should this case get an analyst?"""
        self._roll(day_index)
        if review_value < self.threshold:
            return False
        # Cleared the bar: this is demand, whether or not an analyst is free.
        self._demand_today += 1
        if self._used_today >= self.capacity_per_day:
            return False
        self._used_today += 1
        return True

    @property
    def used_today(self) -> int:
        return self._used_today

    @property
    def demand_today(self) -> int:
        return self._demand_today


# --------------------------------------------------------------------- overrides

@dataclass
class Override:
    name: str
    reason: str
    floor: Decision


# Escalation rules. Each maps a pattern of evidence to a *minimum* action. They can
# only make a decision stricter, never looser: a rule that can silently approve
# something the model flagged is an attack surface, not a safety net.
def evaluate_overrides(
    evidence: Sequence[Evidence], p_fraud: float, amount: float
) -> List[Override]:
    by_tool = {e.tool: e for e in evidence}
    out: List[Override] = []

    # There is deliberately no geolocation override. Impossible travel is one of the
    # strongest signals on real card-present data, but on this benchmark geolocation
    # measures nothing (ROC-AUC 0.494), so a rule built on it would generate false
    # positives and no catches. The check still reports; it does not escalate.
    # See tribunal/tools/heuristics.py:GEO_IS_DIAGNOSTIC.

    vel = by_tool.get("velocity")
    spend = by_tool.get("spend_profile")
    if vel is not None and spend is not None and vel.severity >= 0.6 and spend.severity >= 0.6:
        out.append(Override(
            "burst_and_spike",
            f"{vel.finding} {spend.finding} Testing-then-cashing-out is the classic "
            f"card-not-present pattern.",
            Decision.REVIEW,
        ))

    # A large amount at a merchant this card has never used, at an hour this card is
    # never active, is worth a challenge even at a modest score.
    nov = by_tool.get("novelty")
    tmp = by_tool.get("temporal")
    if (nov is not None and tmp is not None and amount >= 1000
            and nov.severity >= 0.5 and tmp.severity >= 0.5):
        out.append(Override(
            "night_novel_large",
            f"{tmp.finding} {nov.finding} Large first-time spend outside this card's "
            f"normal hours.",
            Decision.STEP_UP,
        ))

    return out


def _at_least(a: Decision, b: Decision) -> Decision:
    """The stricter of two actions, using ACTIONS order as the friction ranking."""
    return a if ACTIONS.index(a) >= ACTIONS.index(b) else b


@dataclass
class PolicyResult:
    decision: Decision
    unconstrained: Decision       # what pure expected-cost minimisation wanted
    costs: Dict[Decision, float]
    review_value: float
    overrides: List[Override]
    budget_blocked: bool          # review was optimal but capacity said no


def decide(
    p_fraud: float,
    amount: float,
    evidence: Sequence[Evidence],
    cm: CostModel,
    budget: Optional[ReviewBudget] = None,
    day_index: int = 0,
) -> PolicyResult:
    costs = expected_costs(p_fraud, amount, cm)

    unconstrained = min(ACTIONS, key=lambda d: costs[d])

    # What review is worth: the saving over the best action available without an
    # analyst. If that is negative, review is not worth a person's time at any price.
    best_without_review = min(
        (d for d in ACTIONS if d is not Decision.REVIEW), key=lambda d: costs[d]
    )
    review_value = costs[best_without_review] - costs[Decision.REVIEW]

    overrides = evaluate_overrides(evidence, p_fraud, amount)

    decision = unconstrained
    for ov in overrides:
        decision = _at_least(decision, ov.floor)

    budget_blocked = False
    if decision is Decision.REVIEW:
        admitted = budget is None or budget.admit(day_index, review_value)
        if not admitted:
            budget_blocked = True
            # No analyst available. Fall back to the best action we can take alone -
            # which for a genuinely risky case is a step-up or a block, not an allow.
            decision = best_without_review
            for ov in overrides:
                if ov.floor is not Decision.REVIEW:
                    decision = _at_least(decision, ov.floor)

    return PolicyResult(
        decision=decision,
        unconstrained=unconstrained,
        costs=costs,
        review_value=review_value,
        overrides=overrides,
        budget_blocked=budget_blocked,
    )
