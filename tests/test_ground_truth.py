"""
test_ground_truth.py  --  Module A: proof the answer key is actually optimal
============================================================================

We do NOT trust the closed-form optima on faith. Each one is re-derived by a second,
independent method and must agree:

    eoq          -- vs a fine GRID SEARCH over order quantity (total-cost curve)
    newsvendor   -- vs a MONTE-CARLO expected-profit search over order quantity
    lp_sourcing  -- vs thousands of RANDOM FEASIBLE allocations (none may beat the LP)

Run:  python tests/test_ground_truth.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from problems import generate_benchmark  # noqa: E402


# --------------------------------------------------------------------------- #
def check_eoq(p, grid_points=200_000, tol=0.01):
    """Grid-search the total-cost curve TC(Q) = D/Q*S + Q/2*H; argmin must match EOQ."""
    D = p.params["annual_demand"]; S = p.params["ordering_cost"]; H = p.params["holding_cost_per_unit_year"]
    eoq = p.ground_truth["order_qty"]
    qs = np.linspace(max(1.0, eoq * 0.2), eoq * 2.0, grid_points)
    tc = D / qs * S + qs / 2.0 * H
    q_star = qs[int(np.argmin(tc))]
    rel = abs(q_star - eoq) / eoq
    assert rel < tol, f"{p.pid}: EOQ grid {q_star:.1f} vs closed-form {eoq:.1f} (rel {rel:.3%})"
    return rel


def check_newsvendor(p, n_draws=200_000, grid=400, tol=0.03, seed=7):
    """Monte-Carlo expected profit over a Q-grid; argmax must match the critical-ratio Q*."""
    cost = p.params["unit_cost"]; price = p.params["sell_price"]; salvage = p.params["salvage_value"]
    mean = p.params["demand_mean"]; std = p.params["demand_std"]
    q_star = p.ground_truth["order_qty"]
    rng = np.random.default_rng(seed)
    demand = rng.normal(mean, std, n_draws)
    qs = np.linspace(max(0.0, q_star - 3 * std), q_star + 3 * std, grid)
    # profit(Q) = price*sold + salvage*leftover - cost*Q,  sold = min(Q, demand)
    sold = np.minimum(qs[:, None], demand[None, :])
    leftover = qs[:, None] - sold
    profit = (price * sold + salvage * leftover - cost * qs[:, None]).mean(axis=1)
    q_mc = qs[int(np.argmax(profit))]
    rel = abs(q_mc - q_star) / max(1.0, q_star)
    assert rel < tol, f"{p.pid}: newsvendor MC {q_mc:.1f} vs formula {q_star:.1f} (rel {rel:.3%})"
    return rel


def check_lp_sourcing(p, n_random=5000, seed=11):
    """No random feasible allocation may cost less than the LP optimum."""
    if not p.ground_truth["feasible"]:
        return 0.0
    costs = np.asarray(p.params["costs"], float)
    caps = np.asarray(p.params["capacities"], float)
    dom = np.asarray(p.params["domestic"], float)
    demand = p.params["demand"]; floor = p.params["dcr_floor"]
    opt = p.ground_truth["total_cost"]
    rng = np.random.default_rng(seed)
    n = len(costs)
    best_random = np.inf
    found = 0
    for _ in range(n_random):
        w = rng.random(n)
        x = w / w.sum() * demand                      # random split summing to demand
        if np.any(x > caps + 1e-6):
            continue
        if dom @ x < floor * demand - 1e-6:
            continue
        found += 1
        best_random = min(best_random, costs @ x)
    # LP optimum must be <= every feasible random allocation (allow tiny numerical slack)
    assert opt <= best_random + 1e-6, f"{p.pid}: random alloc {best_random:.1f} beat LP {opt:.1f}"
    return found


# --------------------------------------------------------------------------- #
def main():
    bench = generate_benchmark(n_per_family=12, seed=42)
    by_family = {}
    for p in bench:
        by_family.setdefault(p.family, []).append(p)

    print(f"Validating ground truth for {len(bench)} problems...\n")

    eoq_rel = [check_eoq(p) for p in by_family["eoq"]]
    print(f"  eoq         : {len(eoq_rel):3d} problems  OK   (max grid-vs-formula error {max(eoq_rel):.4%})")

    nv_rel = [check_newsvendor(p) for p in by_family["newsvendor"]]
    print(f"  newsvendor  : {len(nv_rel):3d} problems  OK   (max MC-vs-formula error     {max(nv_rel):.4%})")

    lp_found = [check_lp_sourcing(p) for p in by_family["lp_sourcing"]]
    print(f"  lp_sourcing : {len(lp_found):3d} problems  OK   (LP beat all random feasible allocations)")

    print("\nALL GROUND-TRUTH CHECKS PASSED.")


if __name__ == "__main__":
    main()
