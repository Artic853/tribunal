"""
Tests for the feature engine.

The first two tests are the ones that matter. Point-in-time correctness is an
invariant that is easy to state, easy to break silently, and impossible to notice
from a metrics dashboard - a leak makes the numbers look *better*, so nothing alerts.

The velocity window tests exist because a real bug shipped through this file:
timestamps were converted with `astype("int64") // 10**9` on a microsecond-resolution
column, which compressed 18 months of data into 47 seconds and made every window
feature meaningless while the model's AUC went *up*. `test_window_boundaries` fails
loudly on that class of mistake.
"""

from __future__ import annotations

import math

import pytest

from tribunal.features import (
    FeatureConfig,
    FeatureEngine,
    haversine_km,
)

HOUR = 3600
DAY = 86400
BASE = 1_700_000_000


def txn(ts, cc="c1", amt=100.0, merchant="m1", category="grocery_pos",
        lat=19.07, lon=72.87, mlat=19.07, mlon=72.87, is_fraud=0, city_pop=1000,
        hour=None, dow=2, age=35.0):
    import datetime as dt

    if hour is None:
        hour = dt.datetime.utcfromtimestamp(ts).hour
    return {
        "unix_ts": ts, "cc_num": cc, "amt": amt, "merchant": merchant,
        "category": category, "lat": lat, "long": lon,
        "merch_lat": mlat, "merch_long": mlon, "city_pop": city_pop,
        "hour": hour, "dow": dow, "age_years": age, "is_fraud": is_fraud,
    }


# --------------------------------------------------------------- leakage tests

def same(a, b) -> bool:
    """Dict equality that treats NaN as equal to NaN. Missing history is encoded as
    NaN throughout, and `nan != nan` would make every comparison below vacuous."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                continue
        if x != y:
            return False
    return True


def test_features_depend_only_on_the_past():
    """The feature vector for transaction k must be identical whether or not
    transactions after k exist. This is the whole contract of the engine."""
    stream = [txn(BASE + i * HOUR, amt=100 + i) for i in range(20)]

    e1 = FeatureEngine()
    short = []
    for t in stream[:10]:
        short.append(e1.compute(t))
        e1.observe(t)

    e2 = FeatureEngine()
    long = []
    for t in stream:
        long.append(e2.compute(t))
        e2.observe(t)

    for i, (a, b) in enumerate(zip(short, long)):
        assert same(a, b), f"feature vector {i} changed when future data was added"


def test_future_fraud_does_not_change_past_features():
    """Altering a *label* in the future must not move any earlier feature."""
    def run(future_label):
        eng = FeatureEngine()
        out = []
        for i in range(10):
            t = txn(BASE + i * DAY, is_fraud=future_label if i >= 5 else 0)
            out.append(eng.compute(t))
            eng.observe(t)
        return out[:5]

    for a, b in zip(run(0), run(1)):
        assert same(a, b)


def test_label_delay_holds_labels_back():
    """A confirmed fraud must not influence merchant risk until the dispute window
    has elapsed - at authorisation time nobody knows yet."""
    cfg = FeatureConfig(label_delay_days=7.0)
    eng = FeatureEngine(cfg)

    eng.observe(txn(BASE, merchant="shady", is_fraud=1))

    # Two days later: the label is not yet available.
    f_early = eng.compute(txn(BASE + 2 * DAY, merchant="shady"))
    # Ten days later: it is.
    f_late = eng.compute(txn(BASE + 10 * DAY, merchant="shady"))

    assert f_late["merchant_fraud_rate_sm"] > f_early["merchant_fraud_rate_sm"]


# ------------------------------------------------------------- velocity windows

def test_window_boundaries():
    """Transactions outside a window must not be counted inside it.

    This is the regression test for the microsecond/second conversion bug: with a
    compressed time axis every transaction falls inside every window and these
    assertions fail.
    """
    eng = FeatureEngine()
    # Three transactions: now-2h, now-30min, now-10min
    for offset in (-2 * HOUR, -30 * 60, -10 * 60):
        eng.observe(txn(BASE + offset))

    f = eng.compute(txn(BASE))
    assert f["cnt_1h"] == 2, f"expected 2 transactions in the last hour, got {f['cnt_1h']}"
    assert f["cnt_24h"] == 3
    assert f["cnt_7d"] == 3


def test_windows_are_nested():
    """cnt_1h <= cnt_24h <= cnt_7d, always. Equality across all three on real data
    means the time axis is broken."""
    eng = FeatureEngine()
    # 25 days of history, fed in ascending order as the engine requires.
    for i in reversed(range(200)):
        eng.observe(txn(BASE - i * 3 * HOUR))
    f = eng.compute(txn(BASE + 60))
    assert f["cnt_1h"] <= f["cnt_24h"] <= f["cnt_7d"]
    assert f["cnt_24h"] < f["cnt_7d"], "24h and 7d windows should differ on 25 days of data"


def test_amount_windows_track_counts():
    eng = FeatureEngine()
    eng.observe(txn(BASE - 10 * 60, amt=50.0))
    eng.observe(txn(BASE - 20 * 60, amt=70.0))
    f = eng.compute(txn(BASE))
    assert f["cnt_1h"] == 2
    assert f["amt_1h"] == pytest.approx(120.0)


# ------------------------------------------------------------------- first-seen

def test_first_transaction_has_no_history():
    eng = FeatureEngine()
    f = eng.compute(txn(BASE))
    assert f["card_txn_count"] == 0
    assert f["is_new_merchant"] == 1.0
    assert math.isnan(f["secs_since_last"])


def test_novelty_flips_after_first_use():
    eng = FeatureEngine()
    t = txn(BASE, merchant="m1", category="travel")
    eng.compute(t)
    eng.observe(t)
    f = eng.compute(txn(BASE + HOUR, merchant="m1", category="travel"))
    assert f["is_new_merchant"] == 0.0
    assert f["is_new_category"] == 0.0


def test_cards_are_isolated():
    """State must never bleed between cards."""
    eng = FeatureEngine()
    for i in range(10):
        eng.observe(txn(BASE + i * 60, cc="cardA"))
    f = eng.compute(txn(BASE + 700, cc="cardB"))
    assert f["card_txn_count"] == 0
    assert f["cnt_1h"] == 0


# -------------------------------------------------------------------- geometry

def test_haversine_known_distance():
    # Mumbai to Delhi is about 1150 km.
    d = haversine_km(19.076, 72.877, 28.644, 77.216)
    assert 1100 < d < 1200


def test_haversine_zero():
    assert haversine_km(12.9, 77.6, 12.9, 77.6) == pytest.approx(0.0, abs=1e-9)
