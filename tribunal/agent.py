"""
The agent.

`Tribunal` is the thing that actually decides. It:

  1. computes point-in-time features for the transaction,
  2. runs every deterministic check, collecting evidence,
  3. asks the calibrated model for a probability,
  4. picks the lowest expected-cost action subject to review capacity,
  5. writes a case memo from the evidence,
  6. appends a hash-chained audit record.

A note on what "agent" means here, because it is the design decision most worth
arguing about. There is no language model in this loop. The orchestration is
deterministic: a fixed set of tools, run in a fixed order, combined by an explicit
cost calculation. That is a deliberate choice for a system that sits in the
authorisation path of a payment:

  * It runs in tens of microseconds, not hundreds of milliseconds. An LLM call in
    the auth path is a latency and availability problem, and payments care about
    p99.
  * It is reproducible. The same transaction and the same state give the same
    decision, every time, which is what makes the audit trail worth having and what
    makes a regression test possible at all.
  * It is cheap enough to run on 100% of traffic rather than a sampled slice.

The language model earns its place where a human would otherwise be writing prose:
rewriting the evidence into a case note for the handful of transactions that reach
an analyst. That is in `memo.py`, off the hot path, and the system degrades to
template memos if it is unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .audit import AuditLog
from .economics import CostModel, Decision
from .features import FeatureConfig, FeatureEngine
from .memo import render_memo
from .policy import PolicyResult, ReviewBudget, decide
from .tools.base import Evidence
from .tools.heuristics import registry
from .tools.model import RiskModel

DAY_S = 86400.0


@dataclass
class Verdict:
    decision: Decision
    p_fraud: float
    evidence: List[Evidence]
    policy: PolicyResult
    memo: str
    latency_us: float
    audit_seq: Optional[int] = None
    audit_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "p_fraud": round(self.p_fraud, 6),
            "unconstrained_decision": self.policy.unconstrained.value,
            "expected_costs": {k.value: round(v, 2) for k, v in self.policy.costs.items()},
            "review_value": round(self.policy.review_value, 2),
            "budget_blocked": self.policy.budget_blocked,
            "overrides": [
                {"name": o.name, "floor": o.floor.value, "reason": o.reason}
                for o in self.policy.overrides
            ],
            "evidence": [e.to_dict() for e in self.evidence],
            "latency_us": round(self.latency_us, 1),
        }


@dataclass
class Tribunal:
    """Full decision pipeline over a stream of transactions."""

    model: RiskModel
    cost_model: CostModel = field(default_factory=CostModel)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    budget: Optional[ReviewBudget] = None
    audit: Optional[AuditLog] = None
    # Audit every decision that costs the customer something, and this fraction of
    # approvals. Logging 100% of approvals is mostly noise; logging none of them
    # makes it impossible to audit for bias in what was let through.
    approval_audit_rate: float = 0.01

    engine: FeatureEngine = field(init=False)
    _epoch: Optional[float] = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.engine = FeatureEngine(self.feature_config)

    # ------------------------------------------------------------------ decide

    def _day_index(self, unix_ts: float) -> int:
        if self._epoch is None:
            self._epoch = unix_ts
        return int((unix_ts - self._epoch) // DAY_S)

    def handle(
        self,
        txn: Dict[str, Any],
        p_fraud: Optional[float] = None,
        write_memo: bool = True,
    ) -> Verdict:
        """Decide on one transaction, then fold it into state.

        `p_fraud` may be supplied by the caller when scores were precomputed in
        batch; the result is identical because the same features feed both paths.
        """
        t0 = time.perf_counter_ns()

        feats = self.engine.compute(txn)
        evidence = registry.run_all(feats)

        if p_fraud is None:
            p_fraud = self.model.predict_one(feats)
        evidence.append(self.model.as_evidence(feats, p_fraud))

        result = decide(
            p_fraud=p_fraud,
            amount=float(txn["amt"]),
            evidence=evidence,
            cm=self.cost_model,
            budget=self.budget,
            day_index=self._day_index(float(txn["unix_ts"])),
        )

        memo = render_memo(txn, p_fraud, evidence, result) if write_memo else ""
        latency = (time.perf_counter_ns() - t0) / 1000.0

        verdict = Verdict(
            decision=result.decision,
            p_fraud=p_fraud,
            evidence=evidence,
            policy=result,
            memo=memo,
            latency_us=latency,
        )

        self._maybe_audit(txn, verdict)

        # Only now does the transaction become part of history.
        self.engine.observe(txn)
        self._n += 1
        return verdict

    # ------------------------------------------------------------------- audit

    def _maybe_audit(self, txn: Dict[str, Any], v: Verdict) -> None:
        if self.audit is None:
            return
        if v.decision is Decision.ALLOW:
            # Deterministic sampling, so a rerun logs the same approvals.
            if self.approval_audit_rate <= 0:
                return
            if (self._n % max(int(1 / self.approval_audit_rate), 1)) != 0:
                return

        payload = {
            "trans_ts": int(txn["unix_ts"]),
            "cc_num_masked": _mask(txn.get("cc_num")),
            "merchant": txn.get("merchant"),
            "category": txn.get("category"),
            "amount": float(txn["amt"]),
            **v.to_dict(),
            "memo": v.memo,
        }
        rec = self.audit.append(payload)
        v.audit_seq = rec.seq
        v.audit_hash = rec.hash


def _mask(cc: Any) -> str:
    """Never write a full PAN to a log, even a local one."""
    s = str(cc)
    return f"****{s[-4:]}" if len(s) >= 4 else "****"
