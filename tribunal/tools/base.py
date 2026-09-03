"""
Tool protocol.

Every check the agent can run is a tool. A tool takes the point-in-time feature
vector and returns a piece of *evidence*: what it looked at, what it concluded, how
strongly, and in one sentence a human can read. Nothing is allowed to influence a
decision without leaving evidence behind - that is what makes the audit trail
complete rather than decorative.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Evidence:
    tool: str
    triggered: bool
    # 0.0 = nothing to see here, 1.0 = as alarming as this check can be.
    severity: float
    finding: str
    data: Dict[str, Any] = field(default_factory=dict)
    latency_us: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "triggered": self.triggered,
            "severity": round(self.severity, 4),
            "finding": self.finding,
            "data": {k: _clean(v) for k, v in self.data.items()},
            "latency_us": round(self.latency_us, 1),
        }


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    return v


ToolFn = Callable[[Dict[str, Any]], Evidence]


class ToolRegistry:
    """Ordered collection of tools, run in sequence with per-tool timing."""

    def __init__(self) -> None:
        self._tools: List[tuple[str, ToolFn]] = []

    def register(self, name: str) -> Callable[[ToolFn], ToolFn]:
        def deco(fn: ToolFn) -> ToolFn:
            self._tools.append((name, fn))
            return fn

        return deco

    def add(self, name: str, fn: ToolFn) -> None:
        self._tools.append((name, fn))

    @property
    def names(self) -> List[str]:
        return [n for n, _ in self._tools]

    def run_all(self, feats: Dict[str, Any]) -> List[Evidence]:
        out: List[Evidence] = []
        for name, fn in self._tools:
            t0 = time.perf_counter_ns()
            ev = fn(feats)
            ev.latency_us = (time.perf_counter_ns() - t0) / 1000.0
            out.append(ev)
        return out


def nan_safe(x: Any, default: float = 0.0) -> float:
    """Missing history is not evidence of innocence, but it is not evidence of guilt
    either. Unknown values fall back to a neutral default at the check level; the
    model sees the NaN directly and can learn from its absence."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0->1 ramp between two thresholds, clamped."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))
