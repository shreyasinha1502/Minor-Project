"""
agent.py  --  Module B: The AI decision-maker
=============================================

Two agents behind one interface:

    MockAgent  -- an offline SIMULATOR used to develop and unit-test the verifier
                  WITHOUT spending on API calls. It reproduces documented LLM failure
                  modes (inverted critical ratio, dropped safety stock, constraint-blind
                  greedy sourcing). It is allowed to see ground truth -- it is NOT the
                  thing under audit's trust boundary; it just plays the role of "an agent
                  that is sometimes wrong in realistic ways".

    LLMAgent   -- the REAL agent. Builds a prompt, calls the Anthropic API, parses the
                  JSON decision. Built ready-to-run; needs ANTHROPIC_API_KEY. Nothing
                  else in the project requires it -- the mock path runs with zero keys.

Both return the SAME AgentResponse, so the verifier and scorer never care which ran.

KEY DESIGN (the headline finding): MockAgent sets its confidence from the SAME
distribution whether it is right or wrong. That is deliberate -- it reproduces the real,
documented result that an LLM's stated confidence barely separates its right answers from
its wrong ones, which is exactly why an EXTERNAL verifier is needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from problems import Problem


# --------------------------------------------------------------------------- #
@dataclass
class AgentResponse:
    pid: str
    family: str
    decision: dict[str, Any]              # the actual numbers (order_qty, allocation, ...)
    intermediates: dict[str, Any]         # the agent's stated working (critical_ratio, z, ...)
    confidence: float                     # agent's self-reported confidence in [0,1]
    rationale: str = ""
    error_mode: str = "none"              # (mock only) which failure was injected, for analysis
    raw: str = ""                         # raw LLM text, if any


# --------------------------------------------------------------------------- #
#  MockAgent
# --------------------------------------------------------------------------- #
# Per-variant probability the mock agent makes an error. Reproduces the documented
# pattern: distractors are handled well, adversarial framing is the hardest slice.
_ERROR_PROB = {"base": 0.30, "distractor": 0.22, "adversarial": 0.55}


class MockAgent:
    name = "mock"

    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def _confidence(self) -> float:
        # Same distribution whether right or wrong -> confidence is (deliberately) uninformative.
        return float(np.clip(self.rng.normal(0.91, 0.03), 0.5, 0.999))

    def answer(self, p: Problem) -> AgentResponse:
        make_error = self.rng.random() < _ERROR_PROB.get(p.variant, 0.3)
        if p.family == "eoq":
            return self._eoq(p, make_error)
        if p.family == "newsvendor":
            return self._newsvendor(p, make_error)
        if p.family == "lp_sourcing":
            return self._lp_sourcing(p, make_error)
        raise ValueError(p.family)

    # ----- EOQ -----
    def _eoq(self, p, err):
        gt = p.ground_truth
        eoq = gt["order_qty"]; rop = gt["reorder_point"]; ss = gt["safety_stock"]; z = gt["z"]
        mode = "none"
        if err:
            mode = self.rng.choice(["no_safety_stock", "eoq_drop_2"])
            if mode == "no_safety_stock":                 # forgets safety stock in the ROP
                dm = p.params["daily_demand_mean"]; L = p.params["lead_time_days"]
                rop = dm * L; ss = 0.0
            else:                                         # drops the 2 in sqrt(2DS/H)
                eoq = eoq / np.sqrt(2.0)
        # (when correct, the simulator returns the exact optimum)
        return AgentResponse(
            p.pid, p.family,
            decision={"order_qty": round(float(eoq), 1), "reorder_point": round(float(rop), 1)},
            intermediates={"eoq": round(float(eoq), 1), "safety_stock": round(float(ss), 1), "z": round(float(z), 3)},
            confidence=self._confidence(),
            rationale="EOQ = sqrt(2DS/H); ROP = lead-time demand + safety stock.",
            error_mode=mode)

    # ----- Newsvendor -----
    def _newsvendor(self, p, err):
        gt = p.ground_truth
        mean = p.params["demand_mean"]; std = p.params["demand_std"]
        cr = gt["critical_ratio"]; z = gt["z"]; q = gt["order_qty"]
        mode = "none"
        if err:
            mode = self.rng.choice(["inverted_cr", "ignore_salvage"])
            if mode == "inverted_cr":                     # orders BELOW mean when CR>0.5 (classic)
                q = mean - z * std
                # intermediates still report the (correct) critical ratio -> internal inconsistency
            else:                                         # treats salvage as 0 -> wrong CR -> wrong q
                from scipy.stats import norm
                cost = p.params["unit_cost"]; price = p.params["sell_price"]
                cr = (price - cost) / price
                z = float(norm.ppf(cr)); q = mean + z * std
        # (when correct, the simulator returns the exact optimum)
        return AgentResponse(
            p.pid, p.family,
            decision={"order_qty": round(float(q), 1)},
            intermediates={"critical_ratio": round(float(cr), 4), "z": round(float(z), 3),
                           "demand_mean": mean},
            confidence=self._confidence(),
            rationale="Critical ratio CR=Cu/(Cu+Co); order Q = mean + z*std, z = inverse-normal(CR).",
            error_mode=mode)

    # ----- LP sourcing -----
    def _lp_sourcing(self, p, err):
        costs = np.asarray(p.params["costs"], float)
        caps = np.asarray(p.params["capacities"], float)
        dom = np.asarray(p.params["domestic"], float)
        demand = p.params["demand"]
        n = len(costs)
        mode = "none"
        if err:
            mode = self.rng.choice(["greedy_cheapest", "ignore_dcr"])
            alloc = np.zeros(n)
            order = np.argsort(costs)                     # cheapest first
            remaining = demand
            for i in order:
                take = min(caps[i], remaining)
                alloc[i] = take; remaining -= take
                if remaining <= 0:
                    break
            # 'greedy_cheapest' and 'ignore_dcr' both ignore the domestic floor; greedy also
            # may ignore nothing else -> typically violates DCR (that's the point).
        else:
            alloc = np.asarray(p.ground_truth["allocation"], float)
        total = float(costs @ alloc)
        used = [i + 1 for i in range(n) if alloc[i] > 1e-6]
        return AgentResponse(
            p.pid, p.family,
            decision={"allocation": [round(float(a), 1) for a in alloc]},
            intermediates={"total_cost": round(total, 1), "suppliers_used": used},
            confidence=self._confidence(),
            rationale="Allocate to minimize cost subject to demand, capacity, and the domestic floor.",
            error_mode=mode)


# --------------------------------------------------------------------------- #
#  LLMAgent  (real Anthropic API -- built ready, run later)
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are a supply-chain planning agent. You are given one decision problem with all "
    "the numbers you need. Think step by step, then output ONLY a JSON object with keys: "
    "decision (the numeric answer), intermediates (your key working values), confidence "
    "(0 to 1), rationale (one sentence). Do not include any text outside the JSON."
)

_DECISION_KEYS = {
    "eoq": '"decision": {"order_qty": <number>, "reorder_point": <number>}, '
           '"intermediates": {"eoq": <number>, "safety_stock": <number>, "z": <number>}',
    "newsvendor": '"decision": {"order_qty": <number>}, '
                  '"intermediates": {"critical_ratio": <number>, "z": <number>}',
    "lp_sourcing": '"decision": {"allocation": [<number per supplier>]}, '
                   '"intermediates": {"total_cost": <number>, "suppliers_used": [<indices>]}',
}


class LLMAgent:
    name = "llm"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic to use --agent llm") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to use --agent llm")
        self.client = anthropic.Anthropic()
        self.model = model

    def answer(self, p: Problem) -> AgentResponse:
        user = (f"{p.prompt}\n\nReturn JSON exactly in this shape:\n"
                f"{{{_DECISION_KEYS[p.family]}, \"confidence\": <0-1>, \"rationale\": \"...\"}}")
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024, system=_SYSTEM,
            messages=[{"role": "user", "content": user}])
        text = msg.content[0].text.strip()
        data = _extract_json(text)
        return AgentResponse(
            p.pid, p.family,
            decision=data.get("decision", {}),
            intermediates=data.get("intermediates", {}),
            confidence=float(data.get("confidence", 0.5)),
            rationale=data.get("rationale", ""),
            raw=text)


def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of an LLM reply, tolerating stray prose/fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def get_agent(kind: str, **kw):
    return LLMAgent(**kw) if kind == "llm" else MockAgent(**kw)


if __name__ == "__main__":
    from problems import generate_benchmark
    agent = MockAgent()
    for p in generate_benchmark(n_per_family=1):
        r = agent.answer(p)
        print(f"[{p.pid:28s}] conf={r.confidence:.3f}  err={r.error_mode:16s}  decision={r.decision}")
