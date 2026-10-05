"""
verifier.py  --  Module C (YASHI): the trust / verification layer
=================================================================

The question this file answers: can we catch the agent's WRONG decisions in production,
where the correct answer does NOT exist?

HARD RULE: the verifier never receives Problem.ground_truth. It sees only:
    - the problem parameters (the same numbers the agent saw), and
    - the agent's response (its decision + its stated working + its confidence).
That is exactly what a real deployment has. Ground truth is used later, once, only to
SCORE how well the verifier did (see evaluate.py).

It is NOT another AI. It is five common-sense checks -- think "an auditor for the agent":

    1. Structural       -- is the answer even a valid answer? (present, finite, non-negative)
    2. Feasibility      -- does it break a hard rule? (capacity / demand / domestic-content floor)
    3. Internal         -- does the final number follow from the agent's OWN stated working?
    4. Dominance        -- is the agent beaten by a dumb, naive policy? then it is wrong.
    5. Calibration      -- a HIGH-confidence answer that also tripped a check is the dangerous one.

Each check returns a severity in [0,1]. The risk score is the MAX, so one hard violation
cannot be averaged away by passing checks. Output: ACCEPT / ESCALATE / REJECT.

(A 6th layer -- robustness under input perturbation -- is phase 2 / final review.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

from agent import AgentResponse
from problems import Problem


# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    severity: float          # 0 = looks fine, 1 = certain error
    passed: bool
    detail: str


@dataclass
class VerifierReport:
    pid: str
    family: str
    checks: list[Check] = field(default_factory=list)
    risk_score: float = 0.0          # max severity (after calibration)
    decision: str = "ACCEPT"         # ACCEPT | ESCALATE | REJECT
    confidence: float = 0.0

    def triggered(self) -> list[str]:
        return [c.name for c in self.checks if c.severity > 0.3]


# Decision thresholds. (Phase 2 tunes these on a held-out set; here they are stated plainly.)
ACCEPT_BELOW = 0.33
REJECT_ABOVE = 0.66


# --------------------------------------------------------------------------- #
#  Layer 1: Structural -- is the decision a valid answer at all?
# --------------------------------------------------------------------------- #
def _check_structural(p: Problem, r: AgentResponse) -> Check:
    d = r.decision
    nums: list[float] = []
    if p.family in ("eoq", "newsvendor"):
        keys = ["order_qty", "reorder_point"] if p.family == "eoq" else ["order_qty"]
        for k in keys:
            if k not in d:
                return Check("structural", 1.0, False, f"missing '{k}'")
            nums.append(d[k])
    elif p.family == "lp_sourcing":
        alloc = d.get("allocation")
        if not alloc or len(alloc) != p.params["n_suppliers"]:
            return Check("structural", 1.0, False, "allocation missing or wrong length")
        nums = list(alloc)
    for v in nums:
        if v is None or not math.isfinite(v):
            return Check("structural", 1.0, False, "non-finite value")
        if v < -1e-6:
            return Check("structural", 1.0, False, f"negative quantity ({v:.1f})")
    return Check("structural", 0.0, True, "well-formed, finite, non-negative")


# --------------------------------------------------------------------------- #
#  Layer 2: Feasibility -- does the decision break a HARD constraint?
#  A violated constraint is a CERTAIN error, provable without knowing the optimum.
# --------------------------------------------------------------------------- #
SERVICE_Z_MARGIN = 0.15   # how far below the target service-z we tolerate before flagging


def _check_feasibility(p: Problem, r: AgentResponse) -> Check:
    if p.family == "eoq":
        # Requirement check: does the reorder point actually deliver the TARGET service level?
        # Achieved service = P(lead-time demand <= ROP). Computable from the problem's own
        # numbers (daily std, lead time) -- no answer key needed. Same idea as the DCR floor.
        rop = r.decision.get("reorder_point")
        if rop is None:
            return Check("feasibility", 1.0, False, "no reorder point to check")
        dm = p.params["daily_demand_mean"]; sd = p.params["daily_demand_std"]
        L = p.params["lead_time_days"]; sl = p.params["service_level"]
        lead_demand = dm * L
        sigma_L = sd * math.sqrt(L)
        achieved_z = (rop - lead_demand) / sigma_L if sigma_L > 0 else 0.0
        req_z = norm.ppf(sl)
        if achieved_z < req_z - SERVICE_Z_MARGIN:
            achieved_sl = norm.cdf(achieved_z)
            return Check("feasibility", 0.9, False,
                         f"ROP delivers only {achieved_sl:.0%} service vs {sl:.0%} required "
                         f"(safety stock too low)")
        return Check("feasibility", 0.0, True, "reorder point meets the service-level requirement")

    if p.family != "lp_sourcing":
        return Check("feasibility", 0.0, True, "n/a for this family")
    alloc = np.asarray(r.decision.get("allocation", []), float)
    if alloc.size != p.params["n_suppliers"]:
        return Check("feasibility", 1.0, False, "cannot check: bad allocation shape")
    caps = np.asarray(p.params["capacities"], float)
    dom = np.asarray(p.params["domestic"], float)
    demand = p.params["demand"]; floor = p.params["dcr_floor"]
    problems_found = []
    if abs(alloc.sum() - demand) > 1e-3 * max(1.0, demand):
        problems_found.append(f"demand not met ({alloc.sum():,.0f} vs {demand:,.0f})")
    if np.any(alloc > caps + 1e-3):
        problems_found.append("exceeds a supplier capacity")
    if dom @ alloc < floor * demand - 1e-3:
        got = (dom @ alloc) / demand if demand else 0
        problems_found.append(f"domestic content {got:.0%} < required {floor:.0%} (ALMM/DCR breach)")
    if problems_found:
        return Check("feasibility", 1.0, False, "; ".join(problems_found))
    return Check("feasibility", 0.0, True, "all constraints satisfied")


# --------------------------------------------------------------------------- #
#  Layer 3: Internal consistency -- does the final answer follow from the
#  agent's OWN stated intermediates and the problem's targets?
# --------------------------------------------------------------------------- #
def _check_internal(p: Problem, r: AgentResponse) -> Check:
    im = r.intermediates or {}
    d = r.decision
    if p.family == "newsvendor":
        cr = im.get("critical_ratio")
        q = d.get("order_qty")
        mean = p.params["demand_mean"]
        sigma = p.params["demand_std"]
        material = 0.25 * sigma          # ignore sign-flips too small to matter economically
        if cr is not None and q is not None:
            # CR > 0.5  =>  optimal order is ABOVE the mean. If the agent states CR>0.5
            # but then orders (materially) below the mean, its own numbers contradict each other.
            if cr > 0.5 and q < mean - material:
                return Check("internal", 0.9, False,
                             f"states CR={cr:.2f} (>0.5) yet orders {q:,.0f} < mean {mean:,.0f}")
            if cr < 0.5 and q > mean + material:
                return Check("internal", 0.9, False,
                             f"states CR={cr:.2f} (<0.5) yet orders {q:,.0f} > mean {mean:,.0f}")
    elif p.family == "eoq":
        ss = im.get("safety_stock")
        rop = d.get("reorder_point")
        dm = p.params["daily_demand_mean"]; L = p.params["lead_time_days"]
        sl = p.params["service_level"]
        lead_demand = dm * L
        # A service level above 50% REQUIRES positive safety stock, so ROP must exceed
        # expected lead-time demand. ROP at or below it means safety stock was dropped.
        if rop is not None and rop < lead_demand - 1e-6 * max(1, lead_demand) and sl > 0.5:
            return Check("internal", 0.85, False,
                         f"ROP {rop:,.0f} <= lead-time demand {lead_demand:,.0f}, "
                         f"but {sl:.0%} service needs safety stock")
        if ss is not None and ss <= 1e-6 and sl > 0.5:
            return Check("internal", 0.8, False,
                         f"safety stock ~0 but target service level is {sl:.0%}")
    return Check("internal", 0.0, True, "final answer consistent with stated working")


# --------------------------------------------------------------------------- #
#  Layer 4: Economic dominance -- is the agent beaten by a NAIVE policy?
#  We never recompute the optimum; we just check a dumb baseline doesn't win.
# --------------------------------------------------------------------------- #
def _eoq_total_cost(Q, D, S, H):
    return D / Q * S + Q / 2.0 * H


def _newsvendor_expected_profit(Q, mu, sigma, price, cost, salvage):
    if sigma <= 0:
        sold = min(Q, mu); leftover = max(0.0, Q - mu)
        return price * sold + salvage * leftover - cost * Q
    z = (Q - mu) / sigma
    leftover = (Q - mu) * norm.cdf(z) + sigma * norm.pdf(z)   # E[(Q-D)+]
    sold = mu - (leftover - (Q - mu))                         # E[min(D,Q)]
    return price * sold + salvage * leftover - cost * Q


def _check_dominance(p: Problem, r: AgentResponse) -> Check:
    if p.family == "eoq":
        Q = r.decision.get("order_qty")
        if not Q or Q <= 0:
            return Check("dominance", 0.0, True, "n/a")
        D = p.params["annual_demand"]; S = p.params["ordering_cost"]; H = p.params["holding_cost_per_unit_year"]
        # (a) OPTIMALITY-BALANCE condition: the EOQ optimum is exactly where annual ordering
        # cost (D/Q*S) equals annual holding cost (Q/2*H). That balance is a necessary
        # condition for optimality -- provable from first principles, WITHOUT the answer.
        ordering_rate = D / Q * S
        holding_rate = Q / 2.0 * H
        ratio = max(ordering_rate, holding_rate) / max(1e-9, min(ordering_rate, holding_rate))
        if ratio > 1.3:
            sev = min(0.9, 0.5 * (ratio - 1.0))
            return Check("dominance", max(0.45, sev), False,
                         f"ordering cost (Rs {ordering_rate:,.0f}) and holding cost (Rs {holding_rate:,.0f}) "
                         f"not balanced (ratio {ratio:.2f}) -> order quantity is off the EOQ optimum")
        # (b) naive-baseline check: a sensible EOQ is never beaten by order-monthly/-annual
        agent_tc = _eoq_total_cost(Q, D, S, H)
        best_naive = min(_eoq_total_cost(q, D, S, H) for q in (D / 12.0, D / 4.0, D))
        if agent_tc > best_naive * 1.001:
            sev = min(0.9, (agent_tc / best_naive - 1.0))
            return Check("dominance", max(0.5, sev), False,
                         f"total cost {agent_tc:,.0f} worse than a naive policy {best_naive:,.0f}")
        return Check("dominance", 0.0, True, "order quantity balances ordering vs holding cost (at EOQ)")

    if p.family == "newsvendor":
        Q = r.decision.get("order_qty")
        if Q is None:
            return Check("dominance", 0.0, True, "n/a")
        mu = p.params["demand_mean"]; sigma = p.params["demand_std"]
        price = p.params["sell_price"]; cost = p.params["unit_cost"]; salvage = p.params["salvage_value"]
        agent_p = _newsvendor_expected_profit(Q, mu, sigma, price, cost, salvage)
        base_p = _newsvendor_expected_profit(mu, mu, sigma, price, cost, salvage)  # order-the-mean
        gap = (base_p - agent_p) / max(1.0, abs(base_p))
        if gap > 0.03:            # materially worse than simply ordering the mean
            return Check("dominance", min(0.9, 0.4 + 3 * (gap - 0.03)), False,
                         f"expected profit {agent_p:,.0f} is {gap:.0%} below order-the-mean {base_p:,.0f}")
        return Check("dominance", 0.0, True, "no better than ordering the mean would require; within noise")

    if p.family == "lp_sourcing":
        alloc = np.asarray(r.decision.get("allocation", []), float)
        costs = np.asarray(p.params["costs"], float)
        if alloc.size != costs.size:
            return Check("dominance", 0.0, True, "n/a")
        agent_cost = float(costs @ alloc)
        base = _feasible_sourcing_baseline(p)
        if base is not None and agent_cost > base * 1.001:
            sev = min(0.9, (agent_cost / base - 1.0))
            return Check("dominance", max(0.5, sev), False,
                         f"cost {agent_cost:,.0f} worse than a simple feasible plan {base:,.0f}")
        return Check("dominance", 0.0, True, "no cheaper naive feasible plan found")

    return Check("dominance", 0.0, True, "n/a")


def _feasible_sourcing_baseline(p: Problem):
    """Greedy FEASIBLE plan: fill the domestic floor with cheapest domestic, rest cheapest overall."""
    costs = np.asarray(p.params["costs"], float)
    caps = np.asarray(p.params["capacities"], float).copy()
    dom = np.asarray(p.params["domestic"], bool)
    demand = p.params["demand"]; floor = p.params["dcr_floor"]
    alloc = np.zeros(len(costs))
    need_dom = floor * demand
    for i in sorted(np.where(dom)[0], key=lambda i: costs[i]):
        if need_dom <= 0:
            break
        take = min(caps[i], need_dom)
        alloc[i] += take; caps[i] -= take; need_dom -= take
    remaining = demand - alloc.sum()
    for i in sorted(range(len(costs)), key=lambda i: costs[i]):
        if remaining <= 0:
            break
        take = min(caps[i], remaining)
        alloc[i] += take; caps[i] -= take; remaining -= take
    if remaining > 1e-3:
        return None            # couldn't build a feasible baseline
    return float(costs @ alloc)


# --------------------------------------------------------------------------- #
#  Layer 5: Calibration -- the dangerous combo is HIGH confidence + a tripped check.
#  This does not find new errors; it ESCALATES the confident-but-flagged ones.
# --------------------------------------------------------------------------- #
def _apply_calibration(report: VerifierReport, confidence: float) -> Check:
    base_risk = max((c.severity for c in report.checks), default=0.0)
    if base_risk > 0.3 and confidence >= 0.85:
        bump = 0.15 * confidence
        return Check("calibration", min(1.0, base_risk + bump), False,
                     f"high confidence ({confidence:.2f}) on a flagged answer -> escalate")
    return Check("calibration", 0.0, True, f"confidence {confidence:.2f}, no escalation")


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def verify(p: Problem, r: AgentResponse) -> VerifierReport:
    report = VerifierReport(pid=p.pid, family=p.family, confidence=r.confidence)
    report.checks = [
        _check_structural(p, r),
        _check_feasibility(p, r),
        _check_internal(p, r),
        _check_dominance(p, r),
    ]
    cal = _apply_calibration(report, r.confidence)
    report.checks.append(cal)

    report.risk_score = max(c.severity for c in report.checks)
    if report.risk_score < ACCEPT_BELOW:
        report.decision = "ACCEPT"
    elif report.risk_score > REJECT_ABOVE:
        report.decision = "REJECT"
    else:
        report.decision = "ESCALATE"
    return report


if __name__ == "__main__":
    from problems import generate_benchmark
    from agent import MockAgent
    agent = MockAgent()
    for p in generate_benchmark(n_per_family=2):
        rep = verify(p, agent.answer(p))
        print(f"[{p.pid:28s}] risk={rep.risk_score:.2f}  {rep.decision:9s}  "
              f"triggered={rep.triggered()}")
