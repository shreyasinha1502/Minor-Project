# Can We Trust the AI Agent? — v2
## Trustworthy AI Agents for Supply Chain Decisions
**Minor Project — BIT Mesra — Team of 3**

---

## 1. The thesis (the "AI revolution in SCM" angle)

AI agents are being sold to run supply-chain planning: what to order, how much, from which
supplier, when. The pitch is autonomy. The blocker to adoption is **trust** — when the agent
is wrong, nothing says so. The output is a confident number in the same format as a correct one.

**Our contribution:** a system that (1) puts an AI agent on real supply-chain decisions that have
*provably optimal* answers, (2) measures how often it is wrong, and (3) builds a **trust / verification
layer** that catches the wrong decisions in production — where the correct answer does NOT exist.

> The revolution of AI in supply chain does not need a smarter agent first. It needs a trust layer.
> This project builds and validates one.

---

## 2. What makes v2 "real" and "bigger" than v1

| | v1 (old) | v2 (this rebuild) |
|---|---|---|
| Agent | MockAgent simulator only | **Real LLM** (Anthropic API) + MockAgent for dev |
| Problem families | 3 (EOQ, newsvendor, LP sourcing) | **6** (add multi-period lot-sizing, 2-echelon base-stock, capacitated DCR sourcing) |
| Scope | single-SKU, single-period | multi-period + multi-echelon + capacity + domestic-content policy |
| Verifier | 5 layers, hand-tuned thresholds | 6 layers + **thresholds tuned on a held-out set** |
| Robustness | planted failure modes only | **one failure mode the verifier was NOT designed for** (honesty test) |
| Output | CSV/JSON | CSV/JSON + **SCM dashboard** (₹ cost impact, service level, decision quality) |
| Scale | n=120, single seed | larger n, **multiple seeds** |

The three loopholes from v1 — "never run on a real LLM", "thresholds maybe rigged",
"only catches its own planted bugs" — are each closed by a specific item above.

---

## 3. The six supply-chain decision families (the benchmark)

Every problem has a **provably optimal answer**, so the agent can be graded exactly — no human annotator.

1. **EOQ** — order quantity + reorder point. Ground truth: Harris/Wilson closed form, ROP from service-level z.
2. **Newsvendor** — single-period order qty. Ground truth: critical ratio + inverse normal CDF.
3. **LP sourcing (capacitated + DCR)** — multi-supplier allocation under a domestic-content floor (ALMM/DCR). Ground truth: scipy HiGHS. *(reuses Cosmic PV policy knowledge)*
4. **Multi-period lot sizing** — when/how much to order over a horizon. Ground truth: Wagner-Whitin optimal.
5. **Two-echelon base-stock** — stock levels across a serial supply chain. Ground truth: base-stock policy.
6. **Transportation / min-cost flow** — route supply to demand at min cost. Ground truth: LP.

*(Families 4–6 are the "bigger" additions. Families 1–3 are the core carried over.)*

---

## 4. The trust / verification layer (the contribution)

Hard rule: the verifier **never sees the correct answer**. It gets only the problem parameters and the
agent's response — exactly what exists in production.

1. **Structural** — parseable, present, finite, correctly signed
2. **Feasibility** — does the decision satisfy capacity, demand, domestic-content floor? A violated constraint is a *certain* error, provable without the optimum.
3. **Internal consistency** — does the final answer follow from the agent's own stated intermediates?
4. **Economic dominance** — is the agent beaten by a trivial baseline (order-the-mean, cheapest-first)? If yes, it's wrong.
5. **Calibration** — high confidence + any triggered check = escalate hardest.
6. **Robustness / sensitivity** *(new)* — perturb inputs slightly; a trustworthy decision shouldn't swing wildly.

Risk score = MAX of checks → **ACCEPT / ESCALATE / REJECT**.

---

## 5. Team split — 3 people

The project divides into three natural modules. Each person **owns, builds, and defends** one.

### Module A — Benchmark & Ground Truth ("the exam with an answer key")
- Builds the 6 problem generators + independent ground-truth validation (grid search / Monte Carlo / LP re-derivation)
- SCM concepts owned: EOQ, newsvendor, lot-sizing, base-stock, sourcing, transportation
- Files: `problems.py`, `tests/test_ground_truth.py`
- **Owner: _______**

### Module B — The AI Agent ("the decision-maker")
- Builds the LLM agent (Anthropic API) + MockAgent, prompt design, the benchmark runner
- Owns: making it "real", prompt engineering, failure-mode analysis
- Files: `agent.py`, `run_benchmark.py`
- **Owner: _______**

### Module C — Trust & Verification + Evaluation/Dashboard ("the governance") ⭐
- Builds the 6-layer verifier, the scoring (agent accuracy + verifier recall/precision/ROC-AUC), the SCM dashboard
- This is the project's contribution and the strongest analyst slice
- Files: `verifier.py`, `evaluate.py`, dashboard
- **Owner: Yashi (recommended)**

---

## 6. The minor presentation — the 50% to show at the review

A minor review typically shows ~50% (phase 1). **The 50% we present = the full pipeline working
end-to-end on the 3 core families (EOQ, newsvendor, sourcing), with MockAgent + a first real-LLM run,
the 5 core verifier layers, and basic results.**

The remaining 50% (families 4–6, the 6th robustness layer, threshold tuning on held-out data,
the full ₹-impact dashboard, multi-seed runs) is **phase 2 → final review.**

Each person presents their module's slice of that 50%:
- **A** presents the benchmark + why the answers are provably correct (demo: the validation tests passing)
- **B** presents the agent + the first real-LLM result (demo: a live run)
- **C / Yashi** presents the trust layer + the headline finding (demo: verifier catching a wrong decision, the confidence-is-useless result)

---

## 7. Folder layout

```
minor project bit mesra/
  PROJECT_PLAN.md            <- this file
  README.md                  <- public-facing project description
  requirements.txt
  src/
    problems.py              (Module A)
    agent.py                 (Module B)
    verifier.py              (Module C)
    evaluate.py              (Module C)
    run_benchmark.py         (Module B)
    dashboard.py             (Module C)
  tests/
    test_ground_truth.py     (Module A)
  results/                   (generated outputs)
```

---

## 8. Honesty guardrails (non-negotiable)

- Numbers from MockAgent are labelled "simulator"; numbers from the LLM run are labelled "real LLM run, n=__".
- We say "we tested an AI agent on supply-chain decisions", NOT "we proved AI fails X% of the time" (until the real run is done and we quote THAT number).
- Each team member claims only their module: "team of 3; I owned the ____ layer."
- Limitations stated openly in the presentation — naming your own gaps first is what reads as credible.
