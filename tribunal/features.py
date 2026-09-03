"""
Point-in-time feature engine.

The single most common way a fraud model is accidentally made to look brilliant is
target leakage: computing a card's "average amount" or a merchant's "fraud rate"
over the *whole* dataset, including the future, and then scoring a transaction with
it. The reported AUC is then partly a measurement of the future.

This engine makes that impossible by construction:

  * State is built by streaming transactions in strict timestamp order.
  * `compute(txn)` reads only state accumulated from transactions that happened
    strictly *before* that transaction.
  * `observe(txn)` folds the transaction into state afterwards.

It also models a detail that matters in real payment risk and is almost always
ignored: **labels arrive late.** You do not know a transaction was fraud at
authorisation time; you find out when the cardholder disputes it, typically days
or weeks later. Any feature derived from labels (merchant fraud rate, card prior
fraud) is therefore only allowed to use labels older than `label_delay_days`.

`eval/leakage_ablation.py` quantifies what this discipline costs in headline
metrics, and what it buys in honesty.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

EARTH_RADIUS_KM = 6371.0

HOUR = 3600.0
DAY = 86400.0

# Velocity windows, in seconds.
W_1H = HOUR
W_24H = DAY
W_7D = 7 * DAY

NIGHT_HOURS = {22, 23, 0, 1, 2, 3}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class FeatureConfig:
    # How long before a fraud label is usable. Models the chargeback / dispute lag.
    label_delay_days: float = 7.0
    # Beta-prior strength for smoothed rate features, in pseudo-observations.
    smoothing_alpha: float = 20.0
    # Global prior fraud rate used by the smoother before evidence accumulates.
    prior_fraud_rate: float = 0.005
    # Cap on the trailing per-card history retained, in seconds.
    history_window_s: float = W_7D
    # If True, label-derived features are disabled entirely (strictest setting).
    disable_label_features: bool = False


@dataclass
class CardState:
    """Trailing state for one card. Holds only what is needed for the features."""

    events: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)
    # (unix_ts, amt, merch_lat, merch_long) within history_window_s

    n: int = 0
    amt_sum: float = 0.0
    amt_sumsq: float = 0.0
    night_n: int = 0

    last_ts: Optional[float] = None
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None

    merchants: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)

    # Label-derived, released only after the delay.
    confirmed_fraud_n: int = 0
    confirmed_n: int = 0

    def prune(self, now: float, window: float) -> None:
        ev = self.events
        cutoff = now - window
        while ev and ev[0][0] < cutoff:
            ev.popleft()

    def window_stats(self, now: float, window: float) -> Tuple[int, float]:
        cutoff = now - window
        cnt = 0
        total = 0.0
        for ts, amt, _, _ in reversed(self.events):
            if ts < cutoff:
                break
            cnt += 1
            total += amt
        return cnt, total

    @property
    def amt_mean(self) -> float:
        return self.amt_sum / self.n if self.n else 0.0

    @property
    def amt_std(self) -> float:
        if self.n < 2:
            return 0.0
        var = (self.amt_sumsq - self.amt_sum**2 / self.n) / (self.n - 1)
        return math.sqrt(max(var, 0.0))


@dataclass
class MerchantState:
    n: int = 0
    confirmed_n: int = 0
    confirmed_fraud_n: int = 0


# Ordered feature names. The model consumes this exact order.
NUMERIC_FEATURES: List[str] = [
    "amt",
    "log_amt",
    "hour",
    "dow",
    "is_night",
    "age_years",
    "log_city_pop",
    "secs_since_last",
    "log_secs_since_last",
    "cnt_1h",
    "cnt_24h",
    "cnt_7d",
    "amt_1h",
    "amt_24h",
    "amt_7d",
    "card_txn_count",
    "amt_over_card_mean",
    "amt_z",
    "card_night_frac",
    "dist_home_merch_km",
    "dist_prev_merch_km",
    "implied_speed_kmh",
    "card_merchant_seen",
    "card_category_seen",
    "is_new_merchant",
    "is_new_category",
    "merchant_txn_count",
    "merchant_fraud_rate_sm",
    "card_fraud_rate_sm",
]

CATEGORICAL_FEATURES: List[str] = ["category"]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features the engine computes but the model is not allowed to use.
#
# `card_fraud_rate_sm` was the second most important feature by permutation
# importance, worth 0.30 PR-AUC on the held-out window. It was removed anyway,
# because it is worth 0.30 PR-AUC of *the wrong thing*.
#
# In this benchmark a card is defrauded at most once. Measured on the test window,
# cards in the top three quintiles of prior confirmed-fraud rate contain exactly
# zero subsequent fraud, while the bottom quintile runs at 2.94%; the feature's
# single-feature ROC-AUC is 0.102, i.e. strongly predictive with the sign reversed.
# The model had learned "this card has been defrauded before, therefore it is safe."
#
# In production the opposite holds: a card with a confirmed compromise is a
# reissued card belonging to a targeted customer, and is at elevated risk. Shipping
# a model that has internalised the inverse would systematically under-protect
# exactly the customers who have already been harmed once. A feature that is only
# accurate because of how the data was manufactured is not a feature.
#
# It stays in the engine because the *quantity* is right and a production deployment
# on real labels would want it - it is the sign learned from this data that is wrong.
EXCLUDED_FROM_MODEL: List[str] = ["card_fraud_rate_sm"]

# What the model actually trains and predicts on.
MODEL_FEATURES: List[str] = [f for f in ALL_FEATURES if f not in EXCLUDED_FROM_MODEL]


class FeatureEngine:
    """Streaming, point-in-time feature computation.

    Usage:
        eng = FeatureEngine(FeatureConfig())
        for txn in transactions_in_time_order:
            feats = eng.compute(txn)   # sees only the past
            eng.observe(txn)           # now fold it in
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.cfg = config or FeatureConfig()
        self.cards: Dict[Any, CardState] = {}
        self.merchants: Dict[str, MerchantState] = {}
        # min-heap of (release_ts, seq, cc_num, merchant, is_fraud)
        self._pending: List[Tuple[float, int, Any, str, int]] = []
        self._seq = 0
        self._delay_s = self.cfg.label_delay_days * DAY

    # ------------------------------------------------------------------ labels

    def _release_labels(self, now: float) -> None:
        """Apply every label whose dispute window has elapsed by `now`."""
        pend = self._pending
        while pend and pend[0][0] <= now:
            _, _, cc, merch, y = heapq.heappop(pend)
            cs = self.cards.get(cc)
            if cs is not None:
                cs.confirmed_n += 1
                cs.confirmed_fraud_n += y
            ms = self.merchants.get(merch)
            if ms is not None:
                ms.confirmed_n += 1
                ms.confirmed_fraud_n += y

    def _smoothed_rate(self, fraud_n: int, total_n: int) -> float:
        a = self.cfg.smoothing_alpha
        p = self.cfg.prior_fraud_rate
        return (fraud_n + a * p) / (total_n + a)

    # ---------------------------------------------------------------- features

    def compute(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Feature vector for `txn` using only strictly-earlier information."""
        now = float(txn["unix_ts"])
        self._release_labels(now)

        cc = txn["cc_num"]
        cs = self.cards.get(cc)
        ms = self.merchants.get(txn["merchant"])

        amt = float(txn["amt"])
        hour = int(txn["hour"])

        f: Dict[str, Any] = {
            "amt": amt,
            "log_amt": math.log1p(max(amt, 0.0)),
            "hour": float(hour),
            "dow": float(txn["dow"]),
            "is_night": 1.0 if hour in NIGHT_HOURS else 0.0,
            "age_years": float(txn["age_years"]),
            "log_city_pop": math.log1p(float(txn["city_pop"])),
            "category": txn["category"],
        }

        if cs is None:
            # First ever transaction on this card. Everything trailing is unknown;
            # NaN is the honest encoding and the model handles it natively.
            f.update(
                secs_since_last=float("nan"),
                log_secs_since_last=float("nan"),
                cnt_1h=0.0,
                cnt_24h=0.0,
                cnt_7d=0.0,
                amt_1h=0.0,
                amt_24h=0.0,
                amt_7d=0.0,
                card_txn_count=0.0,
                amt_over_card_mean=float("nan"),
                amt_z=float("nan"),
                card_night_frac=float("nan"),
                dist_prev_merch_km=float("nan"),
                implied_speed_kmh=float("nan"),
                card_merchant_seen=0.0,
                card_category_seen=0.0,
                is_new_merchant=1.0,
                is_new_category=1.0,
                card_fraud_rate_sm=self._smoothed_rate(0, 0),
            )
        else:
            cs.prune(now, self.cfg.history_window_s)
            c1, a1 = cs.window_stats(now, W_1H)
            c24, a24 = cs.window_stats(now, W_24H)
            c7, a7 = cs.window_stats(now, W_7D)

            dt = now - cs.last_ts if cs.last_ts is not None else float("nan")
            mean = cs.amt_mean
            std = cs.amt_std

            if cs.last_lat is not None:
                d_prev = haversine_km(
                    cs.last_lat, cs.last_lon, float(txn["merch_lat"]), float(txn["merch_long"])
                )
                # Implied travel speed between consecutive card-present-ish events.
                # Physically impossible values are the classic "card is in two
                # places at once" signal.
                speed = d_prev / (dt / HOUR) if dt and dt > 60 else float("nan")
            else:
                d_prev = float("nan")
                speed = float("nan")

            m_seen = cs.merchants.get(txn["merchant"], 0)
            c_seen = cs.categories.get(txn["category"], 0)

            f.update(
                secs_since_last=dt,
                log_secs_since_last=math.log1p(dt) if dt == dt and dt >= 0 else float("nan"),
                cnt_1h=float(c1),
                cnt_24h=float(c24),
                cnt_7d=float(c7),
                amt_1h=a1,
                amt_24h=a24,
                amt_7d=a7,
                card_txn_count=float(cs.n),
                amt_over_card_mean=amt / mean if mean > 0 else float("nan"),
                amt_z=(amt - mean) / std if std > 1e-9 else float("nan"),
                card_night_frac=cs.night_n / cs.n if cs.n else float("nan"),
                dist_prev_merch_km=d_prev,
                implied_speed_kmh=speed,
                card_merchant_seen=float(m_seen),
                card_category_seen=float(c_seen),
                is_new_merchant=0.0 if m_seen else 1.0,
                is_new_category=0.0 if c_seen else 1.0,
                card_fraud_rate_sm=self._smoothed_rate(cs.confirmed_fraud_n, cs.confirmed_n),
            )

        f["dist_home_merch_km"] = haversine_km(
            float(txn["lat"]), float(txn["long"]), float(txn["merch_lat"]), float(txn["merch_long"])
        )

        if ms is None:
            f["merchant_txn_count"] = 0.0
            f["merchant_fraud_rate_sm"] = self._smoothed_rate(0, 0)
        else:
            f["merchant_txn_count"] = float(ms.n)
            f["merchant_fraud_rate_sm"] = self._smoothed_rate(
                ms.confirmed_fraud_n, ms.confirmed_n
            )

        if self.cfg.disable_label_features:
            f["merchant_fraud_rate_sm"] = float("nan")
            f["card_fraud_rate_sm"] = float("nan")

        return f

    # ----------------------------------------------------------------- observe

    def observe(self, txn: Dict[str, Any]) -> None:
        """Fold a transaction into state. Call only after `compute`."""
        now = float(txn["unix_ts"])
        cc = txn["cc_num"]
        cs = self.cards.get(cc)
        if cs is None:
            cs = CardState()
            self.cards[cc] = cs

        amt = float(txn["amt"])
        mlat, mlon = float(txn["merch_lat"]), float(txn["merch_long"])

        cs.events.append((now, amt, mlat, mlon))
        cs.n += 1
        cs.amt_sum += amt
        cs.amt_sumsq += amt * amt
        if int(txn["hour"]) in NIGHT_HOURS:
            cs.night_n += 1
        cs.last_ts = now
        cs.last_lat = mlat
        cs.last_lon = mlon
        cs.merchants[txn["merchant"]] = cs.merchants.get(txn["merchant"], 0) + 1
        cs.categories[txn["category"]] = cs.categories.get(txn["category"], 0) + 1

        ms = self.merchants.get(txn["merchant"])
        if ms is None:
            ms = MerchantState()
            self.merchants[txn["merchant"]] = ms
        ms.n += 1

        # Queue the label for release after the dispute window.
        self._seq += 1
        heapq.heappush(
            self._pending,
            (now + self._delay_s, self._seq, cc, txn["merchant"], int(txn["is_fraud"])),
        )


def stream_features(
    txns: Iterable[Dict[str, Any]], config: Optional[FeatureConfig] = None
) -> Iterator[Dict[str, Any]]:
    """Yield a feature dict per transaction, in order, leak-free."""
    eng = FeatureEngine(config)
    for t in txns:
        feats = eng.compute(t)
        eng.observe(t)
        yield feats
