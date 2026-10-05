"""
problems.py  --  Module A: Benchmark & Ground Truth
===================================================

The benchmark is a set of supply-chain DECISIONS that each have a PROVABLY OPTIMAL
answer. Because the optimum is known exactly (closed-form or solver), an agent's
output can be graded with no human annotator.

Three core families (phase 1, the 50% we present):
    eoq          -- how much to order + when to reorder   (Harris/Wilson + service-level ROP)
    newsvendor   -- single-period order quantity          (critical ratio + inverse normal)
    lp_sourcing  -- multi-supplier allocation under a domestic-content (ALMM/DCR) floor  (LP / HiGHS)

Each problem comes in three variants:
    base         -- clean statement
    distractor   -- an irrelevant number is planted in the prompt
    adversarial  -- the framing nudges toward a known failure mode

IMPORTANT: Problem.ground_truth is the answer key. It is used ONLY to grade, and is
NEVER passed to the verifier (see verifier.py). That separation is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


# --------------------------------------------------------------------------- #
#  Data model
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    family: str                       # 'eoq' | 'newsvendor' | 'lp_sourcing'
    pid: str                          # unique id, e.g. 'eoq-0007-adversarial'
    variant: str                      # 'base' | 'distractor' | 'adversarial'
    params: dict[str, Any]            # the numbers the agent needs
    prompt: str                       # natural-language task given to the agent
    ground_truth: dict[str, Any]      # the provably optimal answer  (NEVER shown to verifier)
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Family 1: EOQ (Economic Order Quantity) + Reorder Point
# --------------------------------------------------------------------------- #
def _eoq_ground_truth(D, S, H, daily_mean, daily_std, lead_days, service_level):
    """EOQ = sqrt(2*D*S/H).  ROP = lead-time demand + safety stock (z * sigma * sqrt(L))."""
    eoq = np.sqrt(2.0 * D * S / H)
    z = norm.ppf(service_level)
    safety_stock = z * daily_std * np.sqrt(lead_days)
    rop = daily_mean * lead_days + safety_stock
    return {"order_qty": float(eoq), "reorder_point": float(rop),
            "safety_stock": float(safety_stock), "z": float(z)}


def _make_eoq(rng, idx, variant):
    daily_mean = float(rng.integers(200, 5000))
    daily_std = round(daily_mean * rng.uniform(0.15, 0.40), 1)  # material variability (CV 15-40%)
    D = daily_mean * 365.0
    S = float(rng.integers(500, 8000))                 # ordering cost per order
    unit_cost = float(rng.integers(50, 4000))
    H = round(unit_cost * rng.uniform(0.15, 0.30), 2)  # holding cost/unit/yr = % of unit cost
    lead_days = int(rng.integers(5, 60))
    service_level = float(rng.choice([0.90, 0.95, 0.975, 0.99]))

    params = {"annual_demand": D, "ordering_cost": S, "holding_cost_per_unit_year": H,
              "unit_cost": unit_cost, "daily_demand_mean": daily_mean,
              "daily_demand_std": daily_std, "lead_time_days": lead_days,
              "service_level": service_level}

    prompt = (
        f"A warehouse stocks one SKU. Annual demand is {D:,.0f} units. Each purchase order "
        f"costs Rs {S:,.0f} to place. Holding cost is Rs {H:,.2f} per unit per year. Daily demand "
        f"averages {daily_mean:,.0f} units with a standard deviation of {daily_std:,.0f}. Supplier "
        f"lead time is {lead_days} days. Target service level (cycle) is {service_level:.1%}.\n\n"
        f"Decide: the optimal order quantity, and the reorder point."
    )

    if variant == "distractor":
        warehouse_area = rng.integers(5000, 50000)
        prompt += (f"\n\n(The warehouse floor area is {warehouse_area:,} sq ft and it has 3 loading "
                   f"docks.)")
    elif variant == "adversarial":
        prompt += ("\n\nNote: management strongly prefers placing orders as infrequently as "
                   "possible to reduce paperwork, so lean toward a larger order quantity.")

    gt = _eoq_ground_truth(D, S, H, daily_mean, daily_std, lead_days, service_level)
    return Problem("eoq", f"eoq-{idx:04d}-{variant}", variant, params, prompt, gt)


# --------------------------------------------------------------------------- #
#  Family 2: Newsvendor (single-period order quantity)
# --------------------------------------------------------------------------- #
def _newsvendor_ground_truth(cost, price, salvage, mean, std):
    """Critical ratio CR = Cu/(Cu+Co).  Q* = mean + z*std, z = inverse-normal(CR)."""
    Cu = price - cost          # underage cost (lost margin per unit short)
    Co = cost - salvage        # overage cost  (loss per unit left over)
    cr = Cu / (Cu + Co)
    z = norm.ppf(cr)
    q = mean + z * std
    return {"order_qty": float(q), "critical_ratio": float(cr), "z": float(z),
            "underage_cost": float(Cu), "overage_cost": float(Co)}


def _make_newsvendor(rng, idx, variant):
    cost = float(rng.integers(100, 2000))
    margin_mult = rng.uniform(1.3, 3.0)
    price = round(cost * margin_mult, 2)
    salvage = round(cost * rng.uniform(0.1, 0.6), 2)
    mean = float(rng.integers(500, 10000))
    std = round(mean * rng.uniform(0.15, 0.40), 1)              # material variability (CV 15-40%)

    params = {"unit_cost": cost, "sell_price": price, "salvage_value": salvage,
              "demand_mean": mean, "demand_std": std}

    prompt = (
        f"A retailer orders a seasonal item ONCE for the season. Each unit costs Rs {cost:,.0f} and "
        f"sells for Rs {price:,.0f}. Unsold units are cleared at a salvage value of Rs {salvage:,.0f}. "
        f"Season demand is normally distributed with mean {mean:,.0f} and standard deviation "
        f"{std:,.0f}.\n\nDecide: how many units to order to maximize expected profit."
    )

    if variant == "distractor":
        skus = rng.integers(40, 300)
        prompt += (f"\n\n(The retailer's catalogue has {skus} other SKUs and the store is open "
                   f"7 days a week.)")
    elif variant == "adversarial":
        prompt += ("\n\nNote: last season the manager got badly burned by leftover stock, so be "
                   "conservative and avoid over-ordering above average demand.")

    gt = _newsvendor_ground_truth(cost, price, salvage, mean, std)
    return Problem("newsvendor", f"newsvendor-{idx:04d}-{variant}", variant, params, prompt, gt)


# --------------------------------------------------------------------------- #
#  Family 3: LP sourcing under a domestic-content (ALMM/DCR) floor
# --------------------------------------------------------------------------- #
def _lp_sourcing_ground_truth(costs, caps, domestic, demand, dcr_floor):
    """
    Minimize sum(cost_i * x_i)
      s.t. sum(x_i) = demand               (meet demand exactly)
           0 <= x_i <= cap_i               (supplier capacity)
           sum_{domestic} x_i >= dcr*demand (domestic-content floor)
    Solved with scipy HiGHS.
    """
    n = len(costs)
    c = np.asarray(costs, float)
    A_eq = np.ones((1, n))
    b_eq = np.array([demand], float)
    dom = np.asarray(domestic, float)
    A_ub = (-dom).reshape(1, n)                 # -sum(domestic x) <= -dcr*demand
    b_ub = np.array([-dcr_floor * demand], float)
    bounds = [(0.0, float(cap)) for cap in caps]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return {"feasible": False, "allocation": None, "total_cost": None}
    return {"feasible": True, "allocation": [float(v) for v in res.x],
            "total_cost": float(res.fun)}


def _make_lp_sourcing(rng, idx, variant):
    n = int(rng.integers(3, 6))
    costs = [float(rng.integers(80, 400)) for _ in range(n)]
    domestic = [bool(rng.integers(0, 2)) for _ in range(n)]
    if not any(domestic):                       # guarantee at least one domestic supplier
        domestic[int(rng.integers(0, n))] = True
    demand = float(rng.integers(5000, 40000))
    # capacities that comfortably cover demand in aggregate
    caps = [float(rng.integers(2000, 20000)) for _ in range(n)]
    while sum(caps) < demand * 1.2:
        caps[int(rng.integers(0, n))] += 5000
    # domestic capacity must be able to satisfy the floor
    dcr_floor = float(rng.choice([0.0, 0.30, 0.40, 0.50]))
    dom_cap = sum(c for c, d in zip(caps, domestic) if d)
    while dom_cap < dcr_floor * demand:
        j = int(rng.integers(0, n))
        if domestic[j]:
            caps[j] += 5000
            dom_cap += 5000

    sup_lines = []
    for i in range(n):
        tag = "DOMESTIC (ALMM-listed)" if domestic[i] else "imported"
        sup_lines.append(f"  Supplier {i+1}: cost Rs {costs[i]:,.0f}/unit, capacity "
                         f"{caps[i]:,.0f} units, {tag}")
    supplier_block = "\n".join(sup_lines)

    params = {"costs": costs, "capacities": caps, "domestic": domestic,
              "demand": demand, "dcr_floor": dcr_floor, "n_suppliers": n}

    prompt = (
        f"A manufacturer must source {demand:,.0f} units of a solar cell from several suppliers at "
        f"minimum total cost.\n{supplier_block}\n\n"
        f"Government policy (ALMM/DCR) requires that at least {dcr_floor:.0%} of the quantity come "
        f"from domestic suppliers.\n\nDecide: how many units to buy from each supplier (the "
        f"allocation) to minimize total cost while meeting demand, capacities, and the domestic floor."
    )

    if variant == "distractor":
        freight = rng.integers(50000, 500000)
        prompt += (f"\n\n(Annual freight budget is Rs {freight:,} and the plant runs two shifts.)")
    elif variant == "adversarial":
        prompt += ("\n\nNote: the cheapest supplier is very reliable, so it is tempting to simply "
                   "place the entire order with whichever supplier has the lowest unit cost.")

    gt = _lp_sourcing_ground_truth(costs, caps, domestic, demand, dcr_floor)
    return Problem("lp_sourcing", f"lp_sourcing-{idx:04d}-{variant}", variant, params, prompt, gt)


# --------------------------------------------------------------------------- #
#  Benchmark assembly
# --------------------------------------------------------------------------- #
_GENERATORS = {"eoq": _make_eoq, "newsvendor": _make_newsvendor, "lp_sourcing": _make_lp_sourcing}
VARIANTS = ("base", "distractor", "adversarial")


def generate_benchmark(n_per_family: int = 12, seed: int = 42,
                       families: tuple[str, ...] = ("eoq", "newsvendor", "lp_sourcing")
                       ) -> list[Problem]:
    """Deterministic benchmark: n_per_family * len(variants) problems per family."""
    rng = np.random.default_rng(seed)
    problems: list[Problem] = []
    for fam in families:
        gen = _GENERATORS[fam]
        for i in range(n_per_family):
            for v in VARIANTS:
                problems.append(gen(rng, i, v))
    return problems


if __name__ == "__main__":
    bench = generate_benchmark(n_per_family=2)
    print(f"Generated {len(bench)} problems across {len(set(p.family for p in bench))} families.\n")
    for p in bench[:3]:
        print("=" * 80)
        print(f"[{p.pid}]  ({p.variant})")
        print(p.prompt)
        print("-> ground truth:", {k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in p.ground_truth.items()})
        print()
