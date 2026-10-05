"""
run_benchmark.py  --  Module B: orchestration / CLI
===================================================

Ties the pieces together: generate problems -> agent decides -> verifier audits ->
score -> write results.

Usage
-----
    python src/run_benchmark.py --agent mock --n 12          # offline, no API key
    python src/run_benchmark.py --agent mock --n 12 --seed 7

    # the real run (needs: pip install anthropic ; set ANTHROPIC_API_KEY):
    set ANTHROPIC_API_KEY=sk-ant-...        (Windows)   /   export on mac-linux
    python src/run_benchmark.py --agent llm --model claude-sonnet-4-6 --n 10

Outputs:
    results/runs_<agent>_seed<seed>.csv       (one row per problem)
    results/summary_<agent>_seed<seed>.json   (headline metrics)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import get_agent
from evaluate import score_run
from problems import generate_benchmark
from verifier import verify

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser(description="Can We Trust the AI Agent? -- benchmark runner")
    ap.add_argument("--agent", choices=["mock", "llm"], default="mock")
    ap.add_argument("--n", type=int, default=12, help="problems per family per variant group")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    problems = generate_benchmark(n_per_family=args.n, seed=args.seed)
    agent = get_agent(args.agent, model=args.model) if args.agent == "llm" else get_agent(args.agent)

    print(f"Running {len(problems)} problems through agent='{args.agent}' ...")
    responses, reports = [], []
    for i, p in enumerate(problems, 1):
        r = agent.answer(p)
        responses.append(r)
        reports.append(verify(p, r))
        if args.agent == "llm" and i % 10 == 0:
            print(f"  ...{i}/{len(problems)}")

    result = score_run(problems, responses, reports)
    s = result["summary"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{args.agent}_seed{args.seed}"
    csv_path = os.path.join(RESULTS_DIR, f"runs_{tag}.csv")
    json_path = os.path.join(RESULTS_DIR, f"summary_{tag}.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result["records"][0].keys()))
        w.writeheader(); w.writerows(result["records"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)

    _print_summary(s, args)
    print(f"\nWrote {csv_path}\n      {json_path}")


def _print_summary(s, args):
    def pct(x): return "  n/a " if x != x else f"{x:6.1%}"
    print("\n" + "=" * 64)
    print(f"  RESULTS  (agent='{args.agent}', n={s['n_problems']} problems)")
    print("=" * 64)
    print(f"  Agent accuracy                 {pct(s['agent_accuracy'])}")
    print(f"  Error rate BEFORE verification {pct(s['error_rate_before'])}")
    print(f"  Error rate AMONG ACCEPTED      {pct(s['error_rate_among_accepted'])}   <- the payoff")
    print("  " + "-" * 60)
    print(f"  Verifier recall (errors caught){pct(s['verifier_recall'])}")
    print(f"  Verifier precision             {pct(s['verifier_precision'])}")
    print(f"  ROC-AUC of risk score          {s['roc_auc']:6.3f}" if s['roc_auc'] == s['roc_auc'] else "  ROC-AUC   n/a")
    print(f"  TP={s['true_positives']}  FP={s['false_positives']}  FN={s['false_negatives']}"
          f"   (flagged {s['n_flagged']}, accepted {s['n_accepted']})")
    print("  " + "-" * 60)
    print(f"  Mean confidence WHEN RIGHT     {s['mean_confidence_when_right']:.3f}")
    print(f"  Mean confidence WHEN WRONG     {s['mean_confidence_when_wrong']:.3f}")
    print("  -> confidence barely separates right from wrong = an external verifier is needed.")
    if args.agent == "mock":
        print("\n  NOTE: numbers from the offline simulator. For real-LLM results: --agent llm")


if __name__ == "__main__":
    main()
