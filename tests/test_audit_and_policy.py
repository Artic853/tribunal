"""
Tests for the audit chain, the cost model and the review budget controller.
"""

from __future__ import annotations

import json
import os

import pytest

from tribunal.audit import AuditLog, verify
from tribunal.economics import CostModel, Decision, expected_cost
from tribunal.policy import ReviewBudget, decide, expected_costs


# ------------------------------------------------------------------ audit chain

def test_chain_verifies(tmp_path):
    log = AuditLog(str(tmp_path / "audit.jsonl"))
    for i in range(50):
        log.append({"decision": "allow", "amount": float(i)})
    ok, bad, msg = verify(log.path)
    assert ok, msg
    assert bad is None


def test_tampering_is_detected(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    for i in range(10):
        log.append({"decision": "block", "amount": float(i)})

    lines = open(path).read().strip().split("\n")
    rec = json.loads(lines[4])
    rec["payload"]["amount"] = 999999.0        # quietly change a past decision
    lines[4] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    open(path, "w").write("\n".join(lines) + "\n")

    ok, bad, msg = verify(path)
    assert not ok
    assert bad == 4, msg


def test_deletion_is_detected(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    for i in range(10):
        log.append({"i": i})
    lines = open(path).read().strip().split("\n")
    del lines[3]
    open(path, "w").write("\n".join(lines) + "\n")
    ok, bad, _ = verify(path)
    assert not ok


def test_chain_resumes_across_restarts(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    a = AuditLog(path)
    a.append({"i": 0})
    a.append({"i": 1})
    head = a.head

    b = AuditLog(path)               # new process, same file
    assert b.head == head
    b.append({"i": 2})
    ok, _, _ = verify(path)
    assert ok


def test_pan_is_never_logged(tmp_path):
    from tribunal.agent import _mask
    assert _mask(4532015112830366) == "****0366"
    assert "4532015112830366" not in _mask(4532015112830366)


# ------------------------------------------------------------------- economics

def test_allowing_legitimate_traffic_is_free():
    cm = CostModel()
    assert expected_cost(Decision.ALLOW, 5000.0, 0, cm) == 0.0


def test_allowing_fraud_costs_the_amount_plus_fee():
    cm = CostModel()
    c = expected_cost(Decision.ALLOW, 5000.0, 1, cm)
    assert c == pytest.approx(5000.0 + cm.chargeback_fixed_fee)


def test_blocking_fraud_is_free_but_blocking_good_traffic_is_not():
    cm = CostModel()
    assert expected_cost(Decision.BLOCK, 5000.0, 1, cm) == 0.0
    assert expected_cost(Decision.BLOCK, 5000.0, 0, cm) > 0.0


def test_high_risk_large_amount_is_never_allowed():
    """At a 90% probability of fraud on a large amount, approving must not be the
    cheapest action under any sane cost model."""
    cm = CostModel()
    costs = expected_costs(0.9, 50_000.0, cm)
    assert min(costs, key=costs.get) is not Decision.ALLOW


def test_low_risk_small_amount_is_allowed():
    cm = CostModel()
    costs = expected_costs(0.0005, 200.0, cm)
    assert min(costs, key=costs.get) is Decision.ALLOW


def test_decision_threshold_moves_with_amount():
    """The score at which we stop approving must depend on the amount. A single
    global threshold cannot be right for both a 100-rupee and a 100,000-rupee
    transaction, which is the entire argument for an expected-cost policy."""
    cm = CostModel()

    def approves(p, amt):
        c = expected_costs(p, amt, cm)
        return min(c, key=c.get) is Decision.ALLOW

    p = 0.02
    assert approves(p, 100.0), "2% risk on 100 rupees is not worth intervening on"
    assert not approves(p, 100_000.0), "2% risk on 100,000 rupees is worth intervening on"


# --------------------------------------------------------------- review budget

def test_budget_never_exceeds_capacity_in_a_day():
    b = ReviewBudget(capacity_per_day=10, threshold=0.0)
    admitted = sum(b.admit(0, review_value=1000.0) for _ in range(500))
    assert admitted == 10


def test_budget_resets_each_day():
    b = ReviewBudget(capacity_per_day=5, threshold=0.0)
    for day in range(3):
        got = sum(b.admit(day, 1000.0) for _ in range(100))
        assert got == 5


def test_controller_raises_the_bar_when_over_capacity():
    """Spend the whole quota every day and the admission threshold must climb."""
    b = ReviewBudget(capacity_per_day=10, threshold=10.0)
    start = b.threshold
    for day in range(20):
        for _ in range(100):
            b.admit(day, 10_000.0)
    assert b.threshold > start


def test_low_value_cases_are_not_admitted():
    b = ReviewBudget(capacity_per_day=100, threshold=500.0)
    assert not b.admit(0, review_value=10.0)
    assert b.admit(0, review_value=900.0)


# ---------------------------------------------------------------------- policy

def test_overrides_only_escalate():
    """An escalation rule must never soften a decision the cost model made."""
    from tribunal.tools.base import Evidence

    cm = CostModel()
    ev = [Evidence("geo", True, 1.0, "impossible travel", {})]
    res = decide(p_fraud=0.99, amount=90_000.0, evidence=ev, cm=cm)
    # Cost model already wants to block; a step-up override must not downgrade it.
    assert res.decision is Decision.BLOCK


def test_budget_exhaustion_does_not_silently_approve():
    """When review capacity runs out, a risky transaction must fall back to another
    intervention - not to allow."""
    from tribunal.tools.base import Evidence

    cm = CostModel()
    budget = ReviewBudget(capacity_per_day=0)
    res = decide(p_fraud=0.6, amount=40_000.0, evidence=[], cm=cm, budget=budget)
    assert res.decision is not Decision.ALLOW
