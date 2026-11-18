# Day Checklist

Status tracker. Update a row only when its deliverable exists as a file and the commit is made. Deliverables come from ROADMAP.md; this is a pointer, not a restatement.

| Day | Deliverable | Status |
|---|---|---|
| 1 | Scaffolded repo, docs/project_spec.md, docs/research_questions.md, first commit | done |
| 2 | bench/prompts.jsonl frozen; experiments/baseline/*.csv; experiments/env.txt | not started |
| 3 | specserve/model/ + tests | not started |
| 4 | kv_cache.py + correctness tests | not started |
| 5 | tests/test_overfit.py passing (HARD GATE) | not started |
| 6 | Resumable training loop; docs/compute_budget.md from measured throughput | not started |
| 7 | Draft-A launched; decode/speculative.py v0 + toy tests | not started |
| 8 | experiments/draft_a/ | not started |
| 9 | Engine + tests/test_exact_match.py (fp32 100%) | not started |
| 10 | fp16 mismatch audit + first measured speedup | not started |
| 11 | train/distill.py; distillation launched | not started |
| 12 | experiments/acceptance/*.csv | not started |
| 13 | bench/run_bench.py | not started |
| 14 | experiments/bench/results.csv + plots | not started |
| 15 | docs/analysis.md | not started |
| 16–17 | Stretch or buffer | not started |
| 18 | README draft | not started |
| 19 | Clean-clone reproduction | not started |
| 20 | v1.0 tag, resume line, docs/interview_prep.md | not started |

Gate days: **Day 2** (benchmark frozen), **Day 5** (overfit), **Day 9–10** (fp32 exact match).
