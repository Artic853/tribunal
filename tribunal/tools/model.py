"""
The model, wrapped as a tool.

The score is one piece of evidence among several, not the decision. Wrapping it in
the same Evidence interface as the deterministic checks keeps that honest: the
policy sees a probability, the memo sees a sentence, and the audit log records both
alongside everything else that was consulted.

The evidence sentence deliberately reports the probability *and* how unusual it is,
because "0.83" means nothing to an analyst who does not know that the median
transaction scores 0.0004.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Evidence


class RiskModel:
    """Calibrated fraud probability. Loads the artifact written by train_model.py."""

    def __init__(self, path: str = "artifacts/model.pkl") -> None:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        self.model = bundle["model"]
        self.features: List[str] = bundle["features"]
        self.categorical: List[str] = bundle["categorical"]
        self.train_end = bundle.get("train_end")
        # Reference quantiles of the score distribution, for phrasing evidence in
        # terms an analyst can act on. Filled by `fit_reference`.
        self.reference: Dict[str, float] = bundle.get("reference", {})

    # -------------------------------------------------------------- prediction

    def _frame(self, feats: Dict[str, Any]) -> pd.DataFrame:
        row = {k: feats.get(k) for k in self.features}
        X = pd.DataFrame([row], columns=self.features)
        for c in self.categorical:
            X[c] = X[c].astype("category")
        return X

    def predict_one(self, feats: Dict[str, Any]) -> float:
        return float(self.model.predict_proba(self._frame(feats))[0, 1])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].copy()
        for c in self.categorical:
            X[c] = X[c].astype("category")
        return self.model.predict_proba(X)[:, 1]

    def fit_reference(self, scores: np.ndarray) -> None:
        """Record score quantiles so evidence can say 'top 0.1% of traffic'."""
        qs = [0.5, 0.9, 0.99, 0.999, 0.9999]
        self.reference = {f"q{q}": float(np.quantile(scores, q)) for q in qs}

    def percentile_phrase(self, p: float) -> str:
        r = self.reference
        if not r:
            return ""
        if p >= r.get("q0.9999", float("inf")):
            return "top 0.01% of scored traffic"
        if p >= r.get("q0.999", float("inf")):
            return "top 0.1% of scored traffic"
        if p >= r.get("q0.99", float("inf")):
            return "top 1% of scored traffic"
        if p >= r.get("q0.9", float("inf")):
            return "top 10% of scored traffic"
        return "unremarkable against the score distribution"

    # -------------------------------------------------------------- as a tool

    def as_evidence(self, feats: Dict[str, Any], p: Optional[float] = None) -> Evidence:
        if p is None:
            p = self.predict_one(feats)
        phrase = self.percentile_phrase(p)
        triggered = p >= 0.05
        finding = (
            f"Model puts fraud probability at {p:.2%}"
            + (f" - {phrase}." if phrase else ".")
        )
        return Evidence(
            "model", triggered, min(p * 2.0, 1.0), finding,
            {"p_fraud": p, "percentile_band": phrase},
        )
