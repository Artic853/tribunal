"""
Figures for the README and the pitch.

Palette and mark conventions follow a validated categorical set (blue / orange /
aqua), checked for colourblind separation rather than chosen by eye: worst
all-pairs CVD Delta-E 9.2, normal-vision 24.0. Aqua sits below 3:1 against the
light surface, so every chart here carries direct value labels rather than relying
on the fill alone to be readable.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_policies(out):
    d = json.load(open("reports/eval.json"))
    rows = [r for r in d["results"] if r["policy"] != "decline everything"]
    rows.sort(key=lambda r: r["net_saved_vs_status_quo"])
    names = [r["policy"] for r in rows]
    vals = [r["net_saved_vs_status_quo"] for r in rows]
    colors = [CRITICAL if v < 0 else (BLUE if "Tribunal" in n else AQUA)
              for v, n in zip(vals, names)]

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    bars = ax.barh(names, vals, color=colors, height=0.62, zorder=3)
    span = max(vals) - min(min(vals), 0)
    for b, v in zip(bars, vals):
        # Negative labels sit to the *right* of zero so they cannot collide with
        # the category names on the left.
        if v >= 0:
            x, ha = v + span * 0.012, "left"
        else:
            x, ha = span * 0.012, "left"
        ax.text(x, b.get_y() + b.get_height() / 2, f"{v:,.0f}",
                va="center", ha=ha, fontsize=9, color=INK, fontweight="bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Net rupees saved over 92 days vs approving everything")
    ax.set_title("What each policy is worth\n"
                 f"{d['n_transactions']:,} held-out transactions, "
                 f"{d['window_days']} days", loc="left")
    ax.set_xlim(min(vals) * 1.15 if min(vals) < 0 else -span * 0.05, max(vals) * 1.20)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M")
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_sweep(out):
    d = json.load(open("reports/uncertainty_sweep.json"))
    rungs = sorted(d["rungs"], key=lambda r: r["pr_auc"])
    x = [r["pr_auc"] for r in rungs]
    thr = [r["policies"][0]["net_saved_vs_status_quo"] for r in rungs]
    ec = [r["policies"][1]["net_saved_vs_status_quo"] for r in rungs]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(x, thr, "-o", color=ORANGE, lw=2, ms=8, zorder=3,
            label="Best fixed score threshold")
    ax.plot(x, ec, "-o", color=BLUE, lw=2, ms=8, zorder=4,
            label="Expected-cost policy")
    ax.fill_between(x, thr, ec, color=BLUE, alpha=0.08, zorder=2)

    # Two rungs sit within 0.01 PR-AUC of each other, so labels are staggered
    # vertically rather than overprinted.
    prev_x, step = -1.0, 0
    for xi, a, b in zip(x, thr, ec):
        gain = 100 * (b - a) / max(a, 1)
        step = step + 1 if (xi - prev_x) < 0.05 else 0
        prev_x = xi
        ax.annotate(f"+{gain:,.0f}%", (xi, b), textcoords="offset points",
                    xytext=(0, 13 + 15 * step), ha="center", fontsize=9,
                    color=INK, fontweight="bold")

    ax.set_xlabel("Model quality on the held-out window (PR-AUC)")
    ax.set_ylabel("Net rupees saved")
    ax.set_title("The decision layer matters most when the model is weakest\n"
                 "Labels show the uplift of pricing each action over thresholding "
                 "the score", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, max(ec) * 1.34)
    ax.set_xlim(min(x) - 0.06, max(x) + 0.04)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M")
    )
    style(ax, xgrid=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_importance(out):
    d = json.load(open("reports/importance.json"))
    rows = [r for r in d["importances"] if r["drop"] > 0.0005][:10][::-1]
    names = [r["feature"] for r in rows]
    vals = [r["drop"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.barh(names, vals, color=BLUE, height=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("PR-AUC lost when the feature is shuffled (held-out window)")
    ax.set_title("What the model actually uses\n"
                 "Permutation importance, not split gain", loc="left")
    ax.set_xlim(0, max(vals) * 1.15)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_calibration(out):
    import pandas as pd
    df = pd.read_parquet("data/features_scored.parquet")
    te = df[pd.to_datetime(df["ts"]) > "2024-03-31"]
    p = te["p_fraud"].to_numpy()
    y = te["is_fraud"].to_numpy()

    # Restricted to the range where the policy actually chooses between actions.
    # Below 0.001 every action but "allow" is dominated, so calibration there is
    # irrelevant to any decision - and plotting it on a log axis exaggerates a
    # difference between two numbers that are both, operationally, zero.
    quiet = p < 1e-3
    edges = np.array([1e-3, 1e-2, 0.05, 0.15, 0.35, 0.65, 0.9, 1.0])
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot([0, 1], [0, 1], "--", color=INK2, lw=1.2,
            label="Perfect calibration", zorder=2)
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=9, zorder=4,
            label="Tribunal, isotonic-calibrated")
    for x, yv, n in zip(xs, ys, ns):
        ax.annotate(f"n={n:,}", (x, yv), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color=INK2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Predicted probability of fraud")
    ax.set_ylabel("Observed fraud rate on held-out data")
    ax.set_title("The probabilities mean what they say\n"
                 "Required, because the policy multiplies them by rupees", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        f"The remaining {quiet.mean():.1%} of traffic scores below 0.001\n"
        f"and runs at an observed fraud rate of {y[quiet].mean():.4%}.",
        xy=(0.40, 0.06), fontsize=8.5, color=INK2,
    )
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_sensitivity(out):
    d = json.load(open("reports/sensitivity.json"))
    sweeps = d["sweeps"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # Left: uplift of the expected-cost policy over an oracle-tuned threshold, at
    # every parameter setting tested. The message is that it never crosses zero.
    names = list(sweeps.keys())
    for i, param in enumerate(names):
        rows = [r for r in sweeps[param] if r["uplift_pct"] is not None]
        ys = [i] * len(rows)
        xs = [r["uplift_pct"] for r in rows]
        ax1.scatter(xs, ys, s=54, color=BLUE, zorder=3, alpha=0.85,
                    edgecolors=SURFACE, linewidths=1.4)
        dflt = [r for r in rows if r["is_default"]]
        if dflt:
            ax1.scatter([dflt[0]["uplift_pct"]], [i], s=150, facecolors="none",
                        edgecolors=ORANGE, linewidths=2.2, zorder=4)

    ax1.axvline(0, color=CRITICAL, lw=1.6, zorder=2)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax1.set_xlabel("Expected-cost policy vs best fixed threshold (%)")
    ax1.set_title("The ranking does not depend on my assumptions\n"
                  f"{sum(len(v) for v in sweeps.values())} cost settings; "
                  "orange ring = shipped default", loc="left")
    ax1.set_xlim(-1.2, max(r["uplift_pct"] for v in sweeps.values()
                           for r in v if r["uplift_pct"] is not None) * 1.18)
    ax1.invert_yaxis()
    ax1.annotate("a threshold would win\nanywhere left of this line",
                 xy=(-0.08, -0.42), fontsize=8.5, color=CRITICAL,
                 ha="right", va="center")
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)

    # Right: the capacity-planning answer.
    rc = sweeps["review_cost"]
    xs = [r["value"] for r in rc]
    ys = [r["reviews_wanted"] for r in rc]
    ax2.plot(xs, ys, "-o", color=BLUE, lw=2, ms=9, zorder=3)
    for x, y in zip(xs, ys):
        ax2.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=9, color=INK, fontweight="bold")
    dflt = next(r for r in rc if r["is_default"])
    ax2.scatter([dflt["value"]], [dflt["reviews_wanted"]], s=170, facecolors="none",
                edgecolors=ORANGE, linewidths=2.2, zorder=4)
    ax2.set_xscale("log")
    ax2.set_xticks(xs)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("Fully-loaded cost of one manual review (₹)")
    ax2.set_ylabel("Cases the policy wants reviewed")
    ax2.set_title("When is an analyst worth it?\n"
                  "Below about ₹60 a case, and not above it", loc="left")
    ax2.set_ylim(-8, max(ys) * 1.3)
    ax2.grid(color=GRID, lw=0.8)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    os.makedirs("reports", exist_ok=True)
    fig_sensitivity("reports/sensitivity.png")
    fig_policies("reports/policy_comparison.png")
    fig_sweep("reports/model_quality_sweep.png")
    fig_importance("reports/feature_importance.png")
    fig_calibration("reports/calibration.png")
    print("wrote 4 figures to reports/")


if __name__ == "__main__":
    main()
