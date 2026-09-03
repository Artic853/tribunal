"""
Deterministic checks.

These are the cheap, explainable signals a human analyst would look at first. They
run on every transaction in the authorisation path, so they are pure functions over
the already-computed feature vector: no I/O, no model, microseconds each.

They are deliberately *not* a scoring model. Their job is to produce named, citable
evidence. The model produces a probability. The policy combines them. Keeping those
three things separate is what lets the system explain a decline in a sentence that
is actually true, instead of reverse-engineering a story from a SHAP plot.

Thresholds are calibrated on the training window only (see
`scripts/calibrate_thresholds.py`) and stored in `config/thresholds.json`.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Evidence, ToolRegistry, nan_safe, ramp

registry = ToolRegistry()

# Geolocation is measured, not assumed, to be useful. On this benchmark it is not:
# distance from the cardholder's home has a single-feature ROC-AUC of 0.4942 against
# the fraud label - indistinguishable from a coin flip - because the simulator draws
# merchant coordinates from a fixed radius around each customer independently of
# whether the transaction is fraudulent. Implied travel speed is equally flat
# (median 4,011 km/h for legitimate transactions vs 3,462 for fraud) since
# consecutive merchants are scattered at random rather than following a person.
#
# The check still runs and still produces evidence, because on real card-present
# data impossible travel is one of the strongest signals there is. But it is barred
# from escalating a decision here, because on *this* data it would only be
# manufacturing false positives. See reports/geo_signal.md for the measurement.
GEO_IS_DIAGNOSTIC = False


@registry.register("velocity")
def velocity_check(f: Dict[str, Any]) -> Evidence:
    """Bursts of transactions on one card in a short window.

    A card that normally does two or three transactions a day and suddenly does six
    in an hour is either on a shopping spree or in someone else's hands.
    """
    c1 = nan_safe(f.get("cnt_1h"))
    c24 = nan_safe(f.get("cnt_24h"))
    a1 = nan_safe(f.get("amt_1h"))
    hist = nan_safe(f.get("card_txn_count"))

    sev = max(ramp(c1, 2, 7), ramp(c24, 8, 25))
    triggered = sev > 0.2 and hist >= 5

    if triggered:
        finding = (
            f"{int(c1)} transactions in the last hour and {int(c24)} in the last 24h "
            f"on a card with {int(hist)} prior transactions "
            f"({a1:,.0f} spent in the last hour)."
        )
    else:
        finding = f"Transaction rate normal ({int(c1)}/h, {int(c24)}/24h)."

    return Evidence(
        "velocity", triggered, sev, finding,
        {"cnt_1h": c1, "cnt_24h": c24, "amt_1h": a1, "card_txn_count": hist},
    )


@registry.register("geo")
def geo_anomaly(f: Dict[str, Any]) -> Evidence:
    """Impossible travel and distance from the cardholder's home region.

    The strong signal is not distance, it is *speed*: the card cannot be in Mumbai
    and then in Chennai eleven minutes later. Distance from home alone flags every
    holiday, so it only contributes at the extreme.
    """
    speed = nan_safe(f.get("implied_speed_kmh"), -1.0)
    d_home = nan_safe(f.get("dist_home_merch_km"), -1.0)
    d_prev = nan_safe(f.get("dist_prev_merch_km"), -1.0)

    # 900 km/h is roughly a commercial flight; beyond that is not travel.
    speed_sev = ramp(speed, 400, 1200) if speed >= 0 else 0.0
    home_sev = ramp(d_home, 120, 600) if d_home >= 0 else 0.0
    sev = max(speed_sev, 0.5 * home_sev)
    triggered = sev > 0.25

    if not GEO_IS_DIAGNOSTIC:
        # Report the observation, but contribute no severity: see the note at the
        # top of this module for the measurement that justifies this.
        sev = 0.0

    if speed_sev > 0.25:
        finding = (
            f"Implied travel speed of {speed:,.0f} km/h since the previous "
            f"transaction ({d_prev:,.0f} km away) - physically implausible."
        )
    elif home_sev > 0.5:
        finding = f"Merchant is {d_home:,.0f} km from the cardholder's home region."
    else:
        finding = f"Location consistent with card history ({d_home:,.0f} km from home)."

    return Evidence(
        "geo", triggered, sev, finding,
        {"implied_speed_kmh": speed, "dist_home_merch_km": d_home,
         "dist_prev_merch_km": d_prev},
    )


@registry.register("spend_profile")
def spend_profile(f: Dict[str, Any]) -> Evidence:
    """Amount relative to what this specific card normally spends.

    An absolute rupee threshold punishes wealthy customers and misses fraud on
    low-limit cards. The deviation from the card's own baseline is the signal.
    """
    z = nan_safe(f.get("amt_z"), 0.0)
    ratio = nan_safe(f.get("amt_over_card_mean"), 1.0)
    amt = nan_safe(f.get("amt"))
    hist = nan_safe(f.get("card_txn_count"))

    if hist < 10:
        return Evidence(
            "spend_profile", False, 0.0,
            f"Insufficient card history to establish a spend baseline "
            f"({int(hist)} prior transactions).",
            {"amt": amt, "card_txn_count": hist},
        )

    sev = max(ramp(z, 3.0, 10.0), ramp(ratio, 5.0, 25.0))
    triggered = sev > 0.2

    finding = (
        f"Amount {amt:,.0f} is {ratio:.1f}x this card's average and {z:.1f} standard "
        f"deviations above it."
        if triggered
        else f"Amount {amt:,.0f} is within this card's normal range ({ratio:.1f}x average)."
    )
    return Evidence(
        "spend_profile", triggered, sev, finding,
        {"amt": amt, "amt_z": z, "amt_over_card_mean": ratio, "card_txn_count": hist},
    )


@registry.register("merchant_risk")
def merchant_risk(f: Dict[str, Any]) -> Evidence:
    """Historical confirmed-fraud rate at this merchant.

    Uses only labels that have already survived the dispute window, so this is what
    the system could genuinely have known at authorisation time - not hindsight.
    """
    rate = nan_safe(f.get("merchant_fraud_rate_sm"), -1.0)
    n = nan_safe(f.get("merchant_txn_count"))

    if rate < 0:
        return Evidence("merchant_risk", False, 0.0,
                        "Merchant reputation unavailable (label features disabled).", {})

    sev = ramp(rate, 0.01, 0.10)
    triggered = sev > 0.2 and n >= 50

    finding = (
        f"Merchant has a {rate:.2%} confirmed-fraud rate over {int(n):,} observed "
        f"transactions."
        if triggered
        else f"Merchant fraud rate {rate:.2%} over {int(n):,} transactions - unremarkable."
    )
    return Evidence("merchant_risk", triggered, sev, finding,
                    {"merchant_fraud_rate": rate, "merchant_txn_count": n})


@registry.register("novelty")
def novelty(f: Dict[str, Any]) -> Evidence:
    """First time this card has touched this merchant or category.

    Weak on its own - everyone shops somewhere new eventually - but it is what turns
    a large night-time transaction from odd into suspicious.
    """
    new_m = nan_safe(f.get("is_new_merchant"))
    new_c = nan_safe(f.get("is_new_category"))
    hist = nan_safe(f.get("card_txn_count"))

    if hist < 10:
        return Evidence("novelty", False, 0.0,
                        "Card too new for novelty to be meaningful.", {"card_txn_count": hist})

    sev = 0.35 * new_m + 0.45 * new_c
    triggered = bool(new_c) or bool(new_m)

    parts = []
    if new_m:
        parts.append("merchant never used by this card")
    if new_c:
        parts.append("spend category never used by this card")
    finding = ("First-time " + " and ".join(parts) + ".") if parts else \
        "Merchant and category both familiar to this card."

    return Evidence("novelty", triggered, sev, finding,
                    {"is_new_merchant": new_m, "is_new_category": new_c,
                     "card_merchant_seen": nan_safe(f.get("card_merchant_seen")),
                     "card_category_seen": nan_safe(f.get("card_category_seen"))})


@registry.register("temporal")
def temporal(f: Dict[str, Any]) -> Evidence:
    """Night-time activity on a card that does not normally transact at night."""
    is_night = nan_safe(f.get("is_night"))
    frac = nan_safe(f.get("card_night_frac"), -1.0)
    hour = int(nan_safe(f.get("hour")))

    if not is_night or frac < 0:
        return Evidence("temporal", False, 0.0,
                        f"Transaction at {hour:02d}:00, within normal hours for this card.",
                        {"hour": hour, "card_night_frac": frac})

    # Night transaction on a card that is rarely active at night. The population
    # median night share is 0.23 on the validation window, so the ramp is anchored
    # there rather than at an invented constant: a card well below the median that
    # suddenly transacts at 02:00 is the case worth flagging.
    sev = ramp(0.23 - frac, 0.0, 0.18)
    triggered = sev > 0.3
    finding = (
        f"Transaction at {hour:02d}:00; only {frac:.1%} of this card's history is "
        f"night-time activity."
        if triggered
        else f"Transaction at {hour:02d}:00; this card is regularly active at night "
             f"({frac:.1%} of history)."
    )
    return Evidence("temporal", triggered, sev, finding,
                    {"hour": hour, "card_night_frac": frac})
