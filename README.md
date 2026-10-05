# Can We Trust the AI Agent?
### A trust layer for AI agents making supply-chain decisions
**Minor Project — BIT Mesra — Team of 3**

---

## The problem

AI agents are being sold to run supply-chain planning: what to order, how much, from which
supplier, when. The pitch is autonomy. The blocker to adoption is **trust** — when the agent
is wrong, nothing says so. Its output is a confident number in the same format as a correct one.

Most LLM evaluation leans on human preference or fuzzy rubrics. Operations research does not
need either: EOQ, newsvendor, and linear programming have **provably optimal answers**. That
makes supply chain a rare domain where an agent can be graded exactly, with no annotator.

We use that to ask two questions:
1. How often is an AI agent wrong on decisions with a known optimum?
2. **Can we catch the wrong ones *without* knowing the answer?** ← the contribution

A leaderboard is easy. A detector that works in production, where ground truth does not exist,
is the thing a firm can actually deploy.

---

## What the system does

```
   problem  ──►  AI AGENT  ──►  decision  ──►  TRUST LAYER  ──►  ACCEPT / ESCALATE / REJECT
 (provable                      + stated        (never sees
  optimum)                      working          the answer)
```

- **Benchmark** — supply-chain decisions with provably optimal answers (3 core families; 3 more in phase 2).
- **Agent** — a real LLM (Anthropic API) or an offline simulator for development.
- **Trust layer** — five common-sense checks that catch wrong decisions without the answer key.
- **Scoring** — grades the agent *and* the trust layer, and writes a one-page dashboard.

---

## Results (offline simulator, 108 decisions, seed 42)

| Metric | Value |
|---|---|
| Agent accuracy | 80.6% |
| Error rate **before** the trust layer | 19.4% |
| Error rate **among accepted** answers | **0.0%** |
| Errors caught (recall) | 100% |
| Precision | 95.5% |
| ROC-AUC of the risk score | 0.993 |

**The headline finding:** the agent's mean confidence when **wrong** (0.922) is essentially the
same as when **right** (0.913). Its own confidence carries almost no signal — which is exactly
why an *external* verifier is required.

> These numbers come from `MockAgent`, an offline simulator that reproduces documented failure
> modes (inverted critical ratio, dropped safety stock, constraint-blind sourcing). It lets the
> trust layer be built and tested with no API spend. For real-LLM results, run `--agent llm`.

---

## The trust layer — five checks (never sees the answer)

1. **Structural** — is the answer even valid? (present, finite, non-negative)
2. **Feasibility / requirement** — does it break a hard rule? capacity, demand, the ALMM/DCR
   domestic-content floor, or the target service level. A violated requirement is a *certain* error.
3. **Internal consistency** — does the final number follow from the agent's own stated working?
   (e.g. a stated critical ratio above 0.5 but an order below the mean contradicts itself)
4. **Economic dominance** — is the agent beaten by a naive policy, or off the EOQ cost-balance
   point? Then it is wrong — provable without the optimum.
5. **Calibration** — a high-confidence answer that also tripped a check is the dangerous one, so
   it is escalated hardest.

Each check emits a severity in [0,1]; the risk score is the **maximum**, so one hard violation
cannot be averaged away. Output: **ACCEPT / ESCALATE / REJECT**.

*(A 6th layer — robustness under input perturbation — is phase 2.)*

---

## Run it

```bash
pip install -r requirements.txt

python tests/test_ground_truth.py              # prove the answer key is optimal (108 checks)
python src/run_benchmark.py --agent mock --n 12 # offline, no API key
python src/dashboard.py --agent mock --seed 42  # -> results/dashboard_mock_seed42.png

# the real run (needs an Anthropic API key):
set ANTHROPIC_API_KEY=sk-ant-...                # Windows   (export ... on mac/linux)
python src/run_benchmark.py --agent llm --model claude-sonnet-4-6 --n 10
```

---

## Layout

```
src/problems.py          generators + provably-optimal ground truth      (Module A)
src/agent.py             LLMAgent (real) / MockAgent (offline simulator)  (Module B)
src/verifier.py          the five-check trust layer                       (Module C)
src/evaluate.py          scoring: agent accuracy AND detection quality    (Module C)
src/run_benchmark.py     orchestration / CLI                              (Module B)
src/dashboard.py         one-page visual summary                          (Module C)
tests/test_ground_truth.py  independent re-derivation of every optimum    (Module A)
```

---

## Limitations (stated honestly)

- Results so far are from the **simulator**, not a real LLM. The real-LLM run is the next step
  (the code path is ready: `--agent llm`).
- 100% recall is against the simulator's **known** failure modes. The real test of the trust
  layer is a failure mode it was **not** designed for — a phase-2 experiment.
- The three core families are single-SKU / single-period. Multi-period lot-sizing, two-echelon
  base-stock, and transportation are the phase-2 "bigger" families.
- Verifier thresholds are set by judgment; tuning them on a held-out set is phase 2.
- Economic checks need a sensible baseline per family; they do not auto-generalise to new families.

---

## Team

Three modules, each owned, built, and defended by one member:

- **Module A — Benchmark & Ground Truth** — the problem set with a provable answer key.
- **Module B — The AI Agent** — the LLM decision-maker and the benchmark runner.
- **Module C — Trust & Verification + Dashboard** — the five-check verifier, the scoring, and
  the visual summary. *(This is the project's contribution.)*
