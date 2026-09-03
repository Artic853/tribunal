"""
Decision service.

A thin HTTP layer over `Tribunal.handle`. It exists to make one claim checkable:
that the thing evaluated offline and the thing that would run in production are the
same object, not two implementations that drifted apart.

    uvicorn api.main:app --reload

    POST /decide      decide on one transaction
    GET  /audit/verify  recompute the audit hash chain
    GET  /healthz     liveness + model provenance

Deliberately absent: authentication, rate limiting, request signing, a real queue
for held cases, and any kind of persistence beyond the append-only audit file.
This is a prototype of a decision engine, not of a payments platform, and pretending
otherwise in a demo is how a demo becomes a liability.

State note: the feature engine is in-process and in-memory, so this server is
stateful - it learns each card's history as requests arrive, and two replicas would
disagree. A real deployment puts the trailing card state in a shared low-latency
store (the `CardState` object is deliberately small and serialisable for exactly
that reason) and keeps the rest of the path stateless.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tribunal.agent import Tribunal
from tribunal.audit import AuditLog, verify
from tribunal.economics import CostModel
from tribunal.features import FeatureConfig
from tribunal.policy import ReviewBudget
from tribunal.tools.model import RiskModel

AUDIT_PATH = os.environ.get("TRIBUNAL_AUDIT", "artifacts/audit_api.jsonl")
MODEL_PATH = os.environ.get("TRIBUNAL_MODEL", "artifacts/model.pkl")

app = FastAPI(
    title="Tribunal",
    description="Fraud triage that writes down why.",
    version="1.0.0",
)

_agent: Optional[Tribunal] = None


def agent() -> Tribunal:
    global _agent
    if _agent is None:
        model = RiskModel(MODEL_PATH)
        _agent = Tribunal(
            model=model,
            cost_model=CostModel(),
            feature_config=FeatureConfig(),
            budget=ReviewBudget(capacity_per_day=int(os.environ.get("TRIBUNAL_CAPACITY", 20))),
            audit=AuditLog(AUDIT_PATH),
            approval_audit_rate=0.01,
        )
    return _agent


class Transaction(BaseModel):
    cc_num: str = Field(..., description="Card identifier. Never logged in full.")
    amt: float = Field(..., gt=0)
    merchant: str
    category: str
    merch_lat: float
    merch_long: float
    lat: float = Field(..., description="Cardholder home latitude")
    long: float = Field(..., description="Cardholder home longitude")
    city_pop: int = 10000
    age_years: float = 35.0
    ts: Optional[str] = Field(None, description="ISO 8601; defaults to now")

    def to_txn(self) -> Dict[str, Any]:
        when = datetime.fromisoformat(self.ts) if self.ts else datetime.now(timezone.utc)
        return {
            "unix_ts": when.timestamp(),
            "cc_num": self.cc_num,
            "amt": self.amt,
            "merchant": self.merchant,
            "category": self.category,
            "lat": self.lat, "long": self.long,
            "merch_lat": self.merch_lat, "merch_long": self.merch_long,
            "city_pop": self.city_pop,
            "hour": when.hour, "dow": when.weekday(),
            "age_years": self.age_years,
            # The engine records an outcome for its delayed-label features. At
            # authorisation time nobody knows, so it is always 0 here; real labels
            # arrive later through the disputes feed, not through this endpoint.
            "is_fraud": 0,
        }


class Evidence(BaseModel):
    tool: str
    triggered: bool
    severity: float
    finding: str


class DecisionResponse(BaseModel):
    decision: str
    p_fraud: float
    memo: str
    evidence: List[Evidence]
    expected_costs: Dict[str, float]
    overrides: List[str]
    review_capacity_exhausted: bool
    latency_ms: float
    audit_seq: Optional[int]
    audit_hash: Optional[str]


@app.post("/decide", response_model=DecisionResponse)
def decide(txn: Transaction) -> DecisionResponse:
    t0 = time.perf_counter()
    try:
        v = agent().handle(txn.to_txn())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"decision failed: {exc}") from exc

    return DecisionResponse(
        decision=v.decision.value,
        p_fraud=round(v.p_fraud, 6),
        memo=v.memo,
        evidence=[
            Evidence(tool=e.tool, triggered=e.triggered,
                     severity=round(e.severity, 4), finding=e.finding)
            for e in v.evidence
        ],
        expected_costs={k.value: round(c, 2) for k, c in v.policy.costs.items()},
        overrides=[o.name for o in v.policy.overrides],
        review_capacity_exhausted=v.policy.budget_blocked,
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        audit_seq=v.audit_seq,
        audit_hash=v.audit_hash,
    )


@app.get("/audit/verify")
def audit_verify() -> Dict[str, Any]:
    if not os.path.exists(AUDIT_PATH):
        return {"ok": True, "message": "no decisions recorded yet"}
    ok, bad, msg = verify(AUDIT_PATH)
    return {"ok": ok, "first_bad_record": bad, "message": msg}


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    a = agent()
    return {
        "status": "ok",
        "model_trained_through": a.model.train_end,
        "review_capacity_per_day": a.budget.capacity_per_day if a.budget else None,
        "audit_records": len(a.audit) if a.audit else 0,
        "audit_head": a.audit.head[:16] if a.audit else None,
    }
