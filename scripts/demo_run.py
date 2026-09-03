"""
End-to-end run.

Streams the whole transaction history through the live agent - the same
`Tribunal.handle` the API calls - and produces the artifacts that show the system
actually works rather than merely scores well:

  * a hash-chained audit log of every intervention in the test window,
  * case memos written from real evidence,
  * a latency distribution measured on the real decision path.

Two latency numbers are reported, because they answer different questions:

  * **decision latency** - features, six checks, policy, memo, audit. This is the
    work the system does per transaction on top of scoring.
  * **end-to-end latency including model inference** - measured separately on a
    sample, because scoring one row at a time through scikit-learn carries a fixed
    per-call overhead that dominates and that a real deployment would amortise with
    batching or a compiled model server. Reporting only the fast number would be
    dishonest; reporting only the slow one would be misleading about where the cost
    is.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from tribunal.agent import Tribunal
from tribunal.audit import AuditLog, verify
from tribunal.economics import CostModel, Decision
from tribunal.features import FeatureConfig
from tribunal.memo import wrap
from tribunal.policy import ReviewBudget
from tribunal.tools.model import RiskModel

VALID_END = "2024-03-31"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["unix_ts"] = df["ts"].astype("datetime64[s]").astype("int64")
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["age_years"] = (df["ts"] - pd.to_datetime(df["dob"])).dt.days / 365.25
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txns", default="data/transactions.parquet")
    ap.add_argument("--scored", default="data/features_scored.parquet")
    ap.add_argument("--audit", default="artifacts/audit.jsonl")
    ap.add_argument("--capacity", type=int, default=20)
    ap.add_argument("--latency-sample", type=int, default=2000)
    a = ap.parse_args()

    txns = prepare(pd.read_parquet(a.txns))
    scored = pd.read_parquet(a.scored)
    assert len(txns) == len(scored)
    p_all = scored["p_fraud"].to_numpy()

    if os.path.exists(a.audit):
        os.remove(a.audit)

    model = RiskModel()
    model.fit_reference(p_all)

    audit = AuditLog(a.audit)
    agent = Tribunal(
        model=model,
        cost_model=CostModel(),
        feature_config=FeatureConfig(),
        budget=ReviewBudget(capacity_per_day=a.capacity),
        audit=audit,
        approval_audit_rate=0.002,
    )

    cols = ["unix_ts", "cc_num", "amt", "category", "merchant", "lat", "long",
            "merch_lat", "merch_long", "city_pop", "hour", "dow", "age_years", "is_fraud"]
    records = txns[cols].to_dict("records")
    test_start = pd.Timestamp(VALID_END).timestamp()

    counts = {d.value: 0 for d in Decision}
    latencies = []
    examples = {"caught": [], "false_positive": [], "missed_big": []}

    t0 = time.time()
    for i, txn in enumerate(records):
        in_test = txn["unix_ts"] > test_start
        # Audit only the evaluation window: the earlier period exists to warm the
        # feature engine, and logging 18 months of approvals proves nothing.
        agent.audit = audit if in_test else None

        v = agent.handle(txn, p_fraud=float(p_all[i]), write_memo=in_test)

        if in_test:
            counts[v.decision.value] += 1
            latencies.append(v.latency_us)
            y = int(txn["is_fraud"])
            acted = v.decision is not Decision.ALLOW
            if y and acted and len(examples["caught"]) < 3:
                examples["caught"].append(v.memo)
            elif (not y) and acted and len(examples["false_positive"]) < 3:
                examples["false_positive"].append(v.memo)
            elif y and not acted and txn["amt"] > 400 and len(examples["missed_big"]) < 3:
                examples["missed_big"].append(
                    v.memo or f"Approved {txn['amt']:,.0f} at {txn['merchant']} - fraud, missed."
                )

        if i and i % 250_000 == 0:
            print(f"  {i:,}/{len(records):,}  {time.time() - t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    lat = np.array(latencies)

    # Separate measurement: full path including a single-row model call.
    sample = records[-a.latency_sample:]
    m2 = Tribunal(model=model, budget=ReviewBudget(capacity_per_day=a.capacity))
    m2.engine = agent.engine  # reuse warmed state
    e2e = []
    for txn in sample:
        t = time.perf_counter_ns()
        m2.handle(txn, p_fraud=None, write_memo=False)
        e2e.append((time.perf_counter_ns() - t) / 1000.0)
    e2e = np.array(e2e)

    ok, bad, msg = verify(a.audit)

    stats = {
        "transactions_processed": len(records),
        "wall_clock_s": round(elapsed, 1),
        "throughput_per_s": round(len(records) / elapsed, 0),
        "test_window_decisions": counts,
        "decision_latency_us": {
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
            "p99": round(float(np.percentile(lat, 99)), 1),
            "max": round(float(lat.max()), 1),
        },
        "end_to_end_latency_us_incl_model": {
            "p50": round(float(np.percentile(e2e, 50)), 1),
            "p95": round(float(np.percentile(e2e, 95)), 1),
            "p99": round(float(np.percentile(e2e, 99)), 1),
        },
        "audit_records": len(audit),
        "audit_chain_ok": ok,
        "audit_chain_message": msg,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/runtime.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    with open("reports/sample_memos.txt", "w") as fh:
        for label, items in examples.items():
            for m in items:
                fh.write(f"{'=' * 78}\n{label.upper()}\n{'=' * 78}\n{wrap(m)}\n\n")

    print(json.dumps(stats, indent=2))
    print("\nwrote reports/runtime.json, reports/sample_memos.txt, " + a.audit)


if __name__ == "__main__":
    main()
