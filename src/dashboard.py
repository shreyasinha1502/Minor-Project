"""
dashboard.py  --  Module C (YASHI): the visual summary
======================================================

Turns one benchmark run into a single presentation image: KPI cards + the three
charts that tell the story.

    python src/dashboard.py --agent mock --seed 42
    -> results/dashboard_<agent>_seed<seed>.png

Reads the CSV + summary that run_benchmark.py wrote (run that first).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

NAVY = "#1f2d4d"; GREEN = "#2e8b57"; RED = "#c0392b"; AMBER = "#e59a0c"; GREY = "#7f8c8d"


def _load(tag):
    with open(os.path.join(RESULTS_DIR, f"summary_{tag}.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(RESULTS_DIR, f"runs_{tag}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return summary, rows


def _card(ax, title, value, color, sub=""):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.1), 0.96, 0.82, transform=ax.transAxes,
                               facecolor=color, alpha=0.12, edgecolor=color, lw=1.5))
    ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=9.5, color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.42, value, ha="center", va="center", fontsize=20, fontweight="bold",
            color=color, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.18, sub, ha="center", va="center", fontsize=7.5, color=GREY, transform=ax.transAxes)


def build(tag, agent):
    summary, rows = _load(tag)
    pct = lambda x: "n/a" if (x != x) else f"{x:.0%}"

    fig = plt.figure(figsize=(14, 8.6))
    fig.suptitle("Can We Trust the AI Agent?  —  Trust Layer for Supply-Chain Decisions",
                 fontsize=16, fontweight="bold", color=NAVY, y=0.975)
    fig.text(0.5, 0.935,
             f"agent = {agent}  |  {summary['n_problems']} decisions across EOQ, Newsvendor, "
             f"and ALMM/DCR Sourcing" + ("   (offline simulator)" if agent == "mock" else ""),
             ha="center", fontsize=10, color=GREY)

    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1.3, 1.3], hspace=0.45, wspace=0.25,
                          left=0.06, right=0.96, top=0.88, bottom=0.07)

    # ---- Row 1: KPI cards ----
    _card(fig.add_subplot(gs[0, 0]), "Error rate BEFORE", pct(summary["error_rate_before"]),
          RED, "of the agent's raw decisions")
    _card(fig.add_subplot(gs[0, 1]), "Error rate AMONG ACCEPTED", pct(summary["error_rate_among_accepted"]),
          GREEN, "after the trust layer")
    _card(fig.add_subplot(gs[0, 2]), "Errors caught (recall)", pct(summary["verifier_recall"]),
          NAVY, f"precision {pct(summary['verifier_precision'])}")
    _card(fig.add_subplot(gs[0, 3]), "ROC-AUC of risk score",
          f"{summary['roc_auc']:.3f}" if summary["roc_auc"] == summary["roc_auc"] else "n/a",
          AMBER, "separates right from wrong")

    # ---- Row 2 left: the headline finding -- confidence right vs wrong ----
    ax = fig.add_subplot(gs[1, :2])
    cr = summary["mean_confidence_when_right"]; cw = summary["mean_confidence_when_wrong"]
    ax.bar(["when RIGHT", "when WRONG"], [cr, cw], color=[GREEN, RED], width=0.5)
    for x, v in zip([0, 1], [cr, cw]):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("mean self-reported confidence")
    ax.set_title("The agent is just as confident when it is WRONG\n"
                 "→ its confidence can't police itself → an external verifier is needed",
                 fontsize=10.5, color=NAVY)
    ax.axhline(cr, ls="--", lw=0.8, color=GREY)

    # ---- Row 2 right: risk-score separation (right vs wrong) ----
    ax = fig.add_subplot(gs[1, 2:])
    risk_right = [float(r["risk_score"]) for r in rows if r["is_error"] == "False"]
    risk_wrong = [float(r["risk_score"]) for r in rows if r["is_error"] == "True"]
    bins = np.linspace(0, 1, 16)
    ax.hist(risk_right, bins=bins, color=GREEN, alpha=0.7, label="correct decisions")
    ax.hist(risk_wrong, bins=bins, color=RED, alpha=0.7, label="wrong decisions")
    ax.axvline(0.33, ls="--", color=GREY, lw=1); ax.axvline(0.66, ls="--", color=GREY, lw=1)
    ax.text(0.16, ax.get_ylim()[1] * 0.9, "ACCEPT", ha="center", fontsize=8, color=GREEN)
    ax.text(0.50, ax.get_ylim()[1] * 0.9, "ESCALATE", ha="center", fontsize=8, color=AMBER)
    ax.text(0.83, ax.get_ylim()[1] * 0.9, "REJECT", ha="center", fontsize=8, color=RED)
    ax.set_xlabel("verifier risk score"); ax.set_ylabel("number of decisions")
    ax.set_title("The risk score cleanly separates right from wrong", fontsize=10.5, color=NAVY)
    ax.legend(fontsize=8)

    # ---- Row 3 left: verifier decision mix ----
    ax = fig.add_subplot(gs[2, :2])
    decs = [r["verifier_decision"] for r in rows]
    order = ["ACCEPT", "ESCALATE", "REJECT"]
    counts = [decs.count(d) for d in order]
    ax.bar(order, counts, color=[GREEN, AMBER, RED], width=0.55)
    for x, v in enumerate(counts):
        ax.text(x, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("decisions"); ax.set_title("What the trust layer did with each decision",
                                             fontsize=10.5, color=NAVY)

    # ---- Row 3 right: which check caught the errors ----
    ax = fig.add_subplot(gs[2, 2:])
    layer_hits = {}
    for r in rows:
        if r["is_error"] == "True" and r["triggered"]:
            for layer in r["triggered"].split("|"):
                if layer != "calibration":
                    layer_hits[layer] = layer_hits.get(layer, 0) + 1
    if layer_hits:
        names = list(layer_hits.keys()); vals = [layer_hits[k] for k in names]
        ax.barh(names, vals, color=NAVY)
        for y, v in enumerate(vals):
            ax.text(v + 0.1, y, str(v), va="center", fontsize=9)
    ax.set_xlabel("errors flagged by this check")
    ax.set_title("Which check caught the errors (a caught error can trip more than one)",
                 fontsize=10.5, color=NAVY)

    out = os.path.join(RESULTS_DIR, f"dashboard_{tag}.png")
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="mock")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(f"{args.agent}_seed{args.seed}", args.agent)
