"""
evaluate.py  --  Module C (YASHI): scoring
==========================================

Two different things get scored, and they must not be confused:

  (A) AGENT quality   -- was the agent's decision actually correct? (needs ground truth)
  (B) VERIFIER quality -- did the trust layer catch the wrong ones? (verifier never saw truth)

Ground truth enters ONLY here, in (A), and only to produce the yes/no "was this an error"
label that (B) is then measured against.

Key metrics:
  - agent accuracy
  - verifier recall  (of the agent's errors, what fraction did we flag?)
  - verifier precision (of what we flagged, what fraction were truly errors?)
  - ROC-AUC of the risk score (how well the 0-1 score separates right from wrong)
  - error rate BEFORE verification vs AMONG ACCEPTED answers
  - the calibration finding: mean confidence when right vs when wrong
"""

from __future__ import annotations

from typing import Any

import numpy as np

import math

from agent import AgentResponse
from problems import Problem
from verifier import VerifierReport, _eoq_total_cost, _newsvendor_expected_profit
from scipy.stats import norm


# --------------------------------------------------------------------------- #
#  (A) Was the agent actually correct?
#  Correctness is judged on ECONOMIC / REQUIREMENT quality, not number-proximity:
#  a decision is correct if it achieves (near-)optimal objective AND meets the
#  problem's stated requirements (e.g. the service level). This matches how a
#  planner would actually judge the decision -- and it stops a logically-wrong
#  answer (dropped safety stock, inverted critical ratio) from being scored "right"
#  just because the number happened to land close.
# --------------------------------------------------------------------------- #
def is_agent_correct(p: Problem, r: AgentResponse, tol: float = 0.05) -> bool:
    gt = p.ground_truth
    d = r.decision
    try:
        if p.family == "eoq":
            D = p.params["annual_demand"]; S = p.params["ordering_cost"]; H = p.params["holding_cost_per_unit_year"]
            Q = d["order_qty"]; rop = d["reorder_point"]
            if Q is None or Q <= 0:
                return False
            ok_cost = _eoq_total_cost(Q, D, S, H) <= _eoq_total_cost(gt["order_qty"], D, S, H) * (1 + tol)
            dm = p.params["daily_demand_mean"]; sd = p.params["daily_demand_std"]
            L = p.params["lead_time_days"]; sl = p.params["service_level"]
            sigma_L = sd * math.sqrt(L)
            achieved_z = (rop - dm * L) / sigma_L if sigma_L > 0 else 0.0
            ok_service = achieved_z >= norm.ppf(sl) - 0.15
            return bool(ok_cost and ok_service)
        if p.family == "newsvendor":
            mu = p.params["demand_mean"]; sg = p.params["demand_std"]
            price = p.params["sell_price"]; cost = p.params["unit_cost"]; salvage = p.params["salvage_value"]
            prof_agent = _newsvendor_expected_profit(d["order_qty"], mu, sg, price, cost, salvage)
            prof_opt = _newsvendor_expected_profit(gt["order_qty"], mu, sg, price, cost, salvage)
            return bool(prof_opt - prof_agent <= tol * abs(prof_opt))
        if p.family == "lp_sourcing":
            if not gt["feasible"]:
                return True
            alloc = np.asarray(d["allocation"], float)
            costs = np.asarray(p.params["costs"], float)
            caps = np.asarray(p.params["capacities"], float)
            dom = np.asarray(p.params["domestic"], float)
            demand = p.params["demand"]; floor = p.params["dcr_floor"]
            # must be feasible AND within tol of optimal cost
            if abs(alloc.sum() - demand) > 1e-3 * demand: return False
            if np.any(alloc > caps + 1e-3): return False
            if dom @ alloc < floor * demand - 1e-3: return False
            return bool(float(costs @ alloc) <= gt["total_cost"] * (1 + tol) + 1e-6)
    except (KeyError, TypeError):
        return False
    return False


# --------------------------------------------------------------------------- #
#  (B) + (A) combined scoring over a full run
# --------------------------------------------------------------------------- #
def score_run(problems: list[Problem], responses: list[AgentResponse],
              reports: list[VerifierReport]) -> dict[str, Any]:
    n = len(problems)
    records = []
    for p, r, rep in zip(problems, responses, reports):
        correct = is_agent_correct(p, r)
        is_error = not correct
        flagged = rep.decision in ("ESCALATE", "REJECT")
        records.append({
            "pid": p.pid, "family": p.family, "variant": p.variant,
            "agent_correct": correct, "is_error": is_error,
            "injected_error_mode": r.error_mode,
            "confidence": r.confidence, "risk_score": rep.risk_score,
            "verifier_decision": rep.decision, "flagged": flagged,
            "triggered": "|".join(rep.triggered()),
        })

    errors = [rec for rec in records if rec["is_error"]]
    flagged = [rec for rec in records if rec["flagged"]]
    tp = sum(1 for rec in records if rec["is_error"] and rec["flagged"])
    fp = sum(1 for rec in records if (not rec["is_error"]) and rec["flagged"])
    fn = sum(1 for rec in records if rec["is_error"] and (not rec["flagged"]))

    accepted = [rec for rec in records if rec["verifier_decision"] == "ACCEPT"]
    accepted_errors = [rec for rec in accepted if rec["is_error"]]

    recall = tp / len(errors) if errors else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    # ROC-AUC of risk score vs the is_error label (needs both classes present)
    roc_auc = float("nan")
    labels = np.array([rec["is_error"] for rec in records], dtype=int)
    scores = np.array([rec["risk_score"] for rec in records], dtype=float)
    if labels.min() != labels.max():
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(labels, scores))
        except Exception:
            roc_auc = _roc_auc_fallback(labels, scores)

    conf_right = np.mean([rec["confidence"] for rec in records if not rec["is_error"]]) if (n - len(errors)) else float("nan")
    conf_wrong = np.mean([rec["confidence"] for rec in records if rec["is_error"]]) if errors else float("nan")

    summary = {
        "n_problems": n,
        "agent_accuracy": 1 - len(errors) / n,
        "error_rate_before": len(errors) / n,
        "error_rate_among_accepted": (len(accepted_errors) / len(accepted)) if accepted else float("nan"),
        "verifier_recall": recall,
        "verifier_precision": precision,
        "roc_auc": roc_auc,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "n_flagged": len(flagged), "n_accepted": len(accepted),
        "mean_confidence_when_right": float(conf_right),
        "mean_confidence_when_wrong": float(conf_wrong),
    }
    return {"summary": summary, "records": records}


def _roc_auc_fallback(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U form of ROC-AUC, in case sklearn is unavailable."""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))
