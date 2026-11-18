# Research Questions

Applies to all three questions (rules live in ROADMAP.md, not repeated here): numbers come only from
files in experiments/; speed numbers count only after the fp32 exact-match gate passes; timing is
batch size 1 with warmup, synchronize, 3 repeats, median and spread. All comparisons use the same
frozen benchmark prompts and the same target model. Formulas: docs/project_spec.md, Verified facts rows 1-2.

Acceptance rate (alpha_hat), defined once and used throughout. Per iteration: successes = accepted
draft tokens; trials = accepted tokens plus 1 if a mismatch occurred (the mismatching guess was tested
and failed). alpha_hat = total successes / total trials. Tokens after a mismatch were never tested, so
they are not counted. (Plain accepted/proposed underestimates alpha as k grows.) Tokens per target
pass is counted directly.

Last-iteration rule. The final iteration of every prompt (the one where generation stops, whether by
max_new_tokens or EOS) is excluded from alpha_hat and tokens per pass, because the stopping condition
cut it short, not the draft. It stays in all timing numbers. The same rule applies to prompt-lookup.
The engine logs one row per iteration (proposed, accepted, mismatch yes/no, stopped yes/no) so the
analysis can apply the rule. The number of excluded iterations, and any prompt left with no iterations
after exclusion, is reported.

Intervals. Paired per-prompt bootstrap intervals are computed in bench/analyze.py with pandas and numpy
only, as a small step and not a separate analysis effort. With ~100 prompts over 4 categories (about
25 each if balanced), per-category intervals will be wide, so "no detectable difference" is an expected,
reportable outcome for RQ1 and RQ3 alike.

Any surprising result is checked for a measurement bug before it is reported as a finding.

## RQ1
### Question in my words
If I take my pretrained draft and distill it on TinyLlama's own predictions (Draft-B), does it guess
TinyLlama's next token more often than the pretrained-only draft (Draft-A), and does that turn into
real wall-clock speedup on a T4?

### How it will be measured
- Same prompts, same target, same k grid, same seed. Only the draft differs.
- Primary: mean tokens per target pass and alpha_hat, counted offline on Day 12 (no timing), then
  again inside the Day 13-14 grid. Secondary: decode tokens/sec relative to plain greedy.
- Paired per-prompt differences (Draft-B minus Draft-A): median plus bootstrap interval, overall and
  per category (see Intervals above). A difference counts only if the interval excludes zero;
  otherwise I report "no detectable difference". Expect the overall result to be more decisive than
  the per-category ones.
- A and B share one architecture, so c is the same. By the formula, any speedup gap should come
  from alpha alone. If it doesn't, overhead or noise is hiding something.
- Limits, stated up front: Draft-B is Draft-A plus extra training, so this tests the recipe as built
  (pretrain, then distill), not distillation separated from extra tokens. The distillation data is
  chat-format UltraChat train, so any gain may concentrate in chat Q&A. RQ1 depends on Draft-B; if
  the Day 12 tripwire drops Draft-B, RQ1 is reported as not answered.

### Which experiment/file answers it
- experiments/acceptance/*.csv (Day 12): alpha_hat and tokens per pass, Draft-A vs Draft-B. First answer.
- experiments/bench/results.csv and its plots (Day 14): spec-A vs spec-B rows. Wall-clock answer.
- docs/analysis.md (Day 15): the written conclusion.

### What result would surprise me
Prior, written before any data: Draft-B beats Draft-A, with the clearest gain on chat Q&A.
- Draft-B no better, or worse, than Draft-A on chat Q&A.
- Draft-B clearly worse than Draft-A on non-chat categories (distillation on chat data hurting general text).
- A clear alpha gain with no speedup gain. Same c, so I would look for overhead, noise, or a bug.

## RQ2
### Question in my words
On a T4 at batch size 1, which k gives the best speedup, which kinds of prompts make speculation
slower than plain greedy decoding, and does the formula, fed with my measured alpha and c, predict
the speedup I actually measure?

### How it will be measured
- Grid: spec-A and spec-B (greedy as the reference) x k in {1, 2, 3, 4, 6, 8} x 4 categories.
- Best k: the k with the highest median speedup within the grid, per draft and per category. If the
  best k sits at the edge of the grid, the true optimum is outside it, and I report it that way.
- Slower than greedy: every config whose median decode tokens/sec falls below greedy's by more than
  greedy's measured repeat-to-repeat spread (Day 13). All of them are listed and plotted, none hidden.
- Formula check (Day 15): time one draft step and one target step in the same harness to get c,
  independently of the speedup measurements. Plug c and alpha_hat into the speedup formula and compare
  predicted vs measured speedup per config. Also compare predicted vs observed tokens per pass, which
  tests the independent-acceptance assumption. Every gap is a finding.
- Profile one speculative iteration by part: draft steps, target verification, cache crops, Python overhead.

### Which experiment/file answers it
- experiments/bench/results.csv and plots (Day 14): speedup vs k per draft, alpha per category,
  speedup per category, every config where speculative is slower.
- Day 15 profiling output in experiments/ (exact path set in the Day 13-15 plan): c and the timing breakdown.
- docs/analysis.md (Day 15): predicted vs measured, with all findings including negative ones.

### What result would surprise me
Prior, written before any data: the best k is interior to the grid rather than at an edge and differs
by category, and speculation is slower than greedy in at least one category.
- The best k sits at an edge of the grid for every category.
- Speculation is slower than greedy in every config. That is a valid result under the ROADMAP tripwire,
  but I would have to explain it with alpha vs c, not shrug.
- Measured speedup above the formula's prediction. The formula ignores crop and Python overhead, so
  it should over-predict. I would suspect my measurement of c or alpha first.
- Near-perfect agreement between predicted and measured. I would check that c was not fitted to match.

## RQ3
### Question in my words
Is training my own 36M draft worth it compared with prompt-lookup decoding, which needs no model at
all, and for which task categories does each one win?

### How it will be measured
- Methods: greedy, prompt-lookup, spec-A, spec-B, all implemented in specserve/decode/ with the same
  target and the same timing harness, so differences come from where the draft tokens come from, not
  from the loop around them. If Draft-B is cut, only spec-A is compared.
- Per category, compare decode tokens/sec, end-to-end latency, and tokens per target pass. The last
  separates "guesses well" from "guesses cheaply": prompt-lookup's drafting cost is near zero, so it
  can win on speed with lower acceptance.
- Compare at matched k and at each method's own best k. k is chosen on the benchmark itself, so the
  best-k comparison is slightly optimistic for both; I say so in the analysis.
- A category winner is declared only under the Intervals rule above; otherwise the category is
  reported as "no detectable difference".
- Cross-check the greedy and prompt-lookup numbers against the Day 2 baselines.

### Which experiment/file answers it
- experiments/bench/results.csv and plots (Day 14): method x category rows, speedup per category.
- experiments/baseline/*.csv (Day 2): sanity cross-check of greedy and prompt-lookup.
- docs/analysis.md (Day 15): the written answer per category.

### What result would surprise me
Prior, written before any data: prompt-lookup wins on copy-heavy summarization/extraction, and the
trained draft is competitive only on chat Q&A and open-ended writing.
- The trained draft beats prompt-lookup on summarization/extraction, where the prompt contains the
  answer's n-grams.
- Prompt-lookup gives no speedup at all on summarization/extraction. That points to my implementation
  or the match settings.
- Prompt-lookup is slower than greedy in categories where it finds few matches (search overhead).