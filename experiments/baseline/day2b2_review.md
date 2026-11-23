# Day 2b-2 Kaggle baseline review

Benchmark execution completed. Final speed acceptance is pending Dev review.
The run manifest records `speed_results_accepted = false` and
`requires_dev_review = true`.

- Runner commit `507162a66b2e6bd9ddad2cffc97b6fe2447a57f9`
- Frozen benchmark SHA-256 `f8d753325a1cc4e52d809e311e47f8182474c6c1504f4c29cd977d1f91289c36`
- Target revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128, Transformers 5.16.1
- 100 frozen prompts, batch size 1, one warmup per method, and three measured repeats per prompt per method. All 600 repeats completed.

## Provisional FP16 decode performance

Prompt lookup achieved a median decode speed ratio of **1.320986814067082×**
versus greedy, approximately **1.32×**. For each repeat, decode throughput
was generated tokens after prefill divided by decode seconds. For each
prompt, the median of the three prompt-lookup rates was divided by the
median of the three greedy rates. The reported value is the median of the
100 per-prompt ratios. All 300 decode-rate repeats per method were valid.
The ratios and spreads were recomputed from `timing.csv` and matched
`summary.json`.

Repeat spread is the maximum minus the minimum decode throughput within a
prompt and method. The table summarizes those per-prompt spreads in tokens/sec.

| Method | Minimum spread | Median spread | Maximum spread |
| --- | ---: | ---: | ---: |
| Greedy | 0.06766381012221956 | 0.8093161957042057 | 3.4186134201859275 |
| Prompt lookup | 0.09072127526339813 | 1.161500638550983 | 9.281030680251419 |

## Correctness and review

Both FP32 exact-match gates passed for each method, 20/20 on the initial
subset and 100/100 on the full benchmark. FP16 prompt lookup had six
distinct reproducible divergences from greedy. Each recurred in all three
measured prompt-lookup repeats, accounting for 18 reviewed timed
divergences. These are measurements of FP16 runs with observed output
divergence.

| Prompt | First differing output index (zero-based) | Greedy → lookup token ID | Review |
| --- | ---: | --- | --- |
| `chat_qa-06` | 66 | 29889 → 363 | Near-tie |
| `chat_qa-10` | 100 | 2472 → 848 | Near-tie |
| `chat_qa-18` | 18 | 6345 → 278 | Near-tie |
| `chat_qa-25` | 90 | 367 → 13100 | Diagnostic-backed numerical drift |
| `open_ended_writing-12` | 57 | 1576 → 6716 | Near-tie |
| `code-22` | 58 | 848 → 263 | Near-tie |

For `chat_qa-25`, the prediction followed the retained KV-cache history.
Identical-cache repeats were exact, both rejected-suffix controls passed,
and all three full FP32 logit vectors agreed pairwise at the diagnostic
tolerances and selected token 2125. The case reviews cleared these
divergences for measurement. Final speed acceptance remains a separate review.

## Evidence

External archive `specserve-reviewed-benchmark-507162a.tar.gz`, SHA-256
`8362f28483c60b9b5b7435a69740f796baefae926eb70111d0be109d8604d403`.
Inside the archive, `specserve-reviewed-benchmark-507162a/experiments/`
contains `env.txt` and the baseline `run.json`, `bench.yaml`,
`correctness.csv`, `timing.csv`, and `summary.json`.
`specserve-fp16-review-507162a/` contains `fp16-reviews.approved.json`
and the numerical drift diagnostic under `evidence/`.
