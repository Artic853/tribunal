"""
Case memos.

When a transaction is held for review, an analyst has to read something. When a
merchant asks why their customer was declined, support has to say something. Both
are currently answered, in most systems, by a number.

A memo is written from the evidence the tools actually produced, in the order that
mattered to the decision. It is generated from templates by default, which means it
is deterministic, free, and cannot hallucinate a fact that no tool observed.

An optional LLM pass rewrites the same evidence into more fluent prose. It is
strictly a *rewriter*: it receives the evidence list and the decision, and it never
sees an incentive or an ability to change them. It is also deliberately kept out of
the authorisation path - it runs only for the ~0.1% of transactions that reach a
human queue, where a few hundred milliseconds and a per-call cost are irrelevant.
Putting a language model in the p99 latency budget of a payment authorisation is a
good way to fail an availability review.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence

from .economics import Decision
from .policy import PolicyResult
from .tools.base import Evidence

ACTION_PHRASE = {
    Decision.ALLOW: "Approved",
    Decision.STEP_UP: "Challenged with an additional authentication factor",
    Decision.REVIEW: "Held for manual review",
    Decision.BLOCK: "Declined",
}


def _ranked_evidence(evidence: Sequence[Evidence]) -> List[Evidence]:
    return sorted(
        [e for e in evidence if e.triggered],
        key=lambda e: e.severity,
        reverse=True,
    )


def render_memo(
    txn: Dict[str, Any],
    p_fraud: float,
    evidence: Sequence[Evidence],
    result: PolicyResult,
) -> str:
    """Deterministic case memo built from observed evidence only."""
    fired = _ranked_evidence(evidence)
    amt = float(txn.get("amt", 0.0))

    lines: List[str] = []
    lines.append(
        f"{ACTION_PHRASE[result.decision]}: {amt:,.0f} at {txn.get('merchant', 'unknown merchant')} "
        f"({txn.get('category', 'uncategorised')})."
    )
    lines.append(
        f"Model probability of fraud {p_fraud:.2%}. Expected cost of this action "
        f"{result.costs[result.decision]:,.0f}, versus "
        f"{result.costs[Decision.ALLOW]:,.0f} if approved outright."
    )

    if fired:
        lines.append("")
        lines.append("Evidence:")
        for e in fired:
            lines.append(f"  - [{e.tool}] {e.finding}")
    else:
        lines.append("")
        lines.append(
            "No deterministic check fired. This decision rests on the model score "
            "alone, which is worth an analyst's scepticism."
        )

    if result.overrides:
        lines.append("")
        lines.append("Escalation rules applied:")
        for ov in result.overrides:
            lines.append(f"  - {ov.name}: {ov.reason}")

    if result.budget_blocked:
        lines.append("")
        lines.append(
            "Manual review was the lowest-cost action but review capacity was "
            f"exhausted for the day. Fell back to '{result.decision.value}'. "
            "This case is a candidate for capacity planning, not a model failure."
        )

    quiet = [e for e in evidence if not e.triggered]
    if quiet:
        lines.append("")
        lines.append("Checks that did not fire: " + ", ".join(e.tool for e in quiet) + ".")

    return "\n".join(lines)


# ------------------------------------------------------------------- LLM rewrite

PROMPT = """You are writing a short case note for a payments fraud analyst.

Rewrite the evidence below into two or three plain sentences explaining why this
transaction was {action}. Rules:
- Use only facts present in the evidence. Do not add, infer, or estimate anything.
- Do not argue for a different decision. You are documenting, not deciding.
- No preamble. Start with the finding that matters most.

Decision: {action}
Model probability of fraud: {p:.2%}
Amount: {amt:,.0f} at {merchant} ({category})

Evidence:
{evidence}
"""


def llm_memo(
    txn: Dict[str, Any],
    p_fraud: float,
    evidence: Sequence[Evidence],
    result: PolicyResult,
    client: Optional[Any] = None,
    model: str = "claude-sonnet-4-5",
) -> str:
    """Optional fluent rewrite. Falls back to the deterministic memo when no API
    client or key is configured, so the system never depends on it being available."""
    fired = _ranked_evidence(evidence)
    if client is None:
        try:
            import anthropic  # type: ignore

            if not os.environ.get("ANTHROPIC_API_KEY"):
                return render_memo(txn, p_fraud, evidence, result)
            client = anthropic.Anthropic()
        except Exception:
            return render_memo(txn, p_fraud, evidence, result)

    body = "\n".join(f"- [{e.tool}] {e.finding}" for e in fired) or "- no checks fired"
    prompt = PROMPT.format(
        action=result.decision.value,
        p=p_fraud,
        amt=float(txn.get("amt", 0.0)),
        merchant=txn.get("merchant", "unknown"),
        category=txn.get("category", "uncategorised"),
        evidence=body,
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        # A memo service outage must never affect a payment decision.
        return render_memo(txn, p_fraud, evidence, result)


def wrap(text: str, width: int = 88) -> str:
    return "\n".join(
        textwrap.fill(line, width) if len(line) > width else line
        for line in text.split("\n")
    )
