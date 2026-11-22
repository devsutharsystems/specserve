# Day 2b-2 Kaggle baselines

Run after Dev approves GPU execution and pushes the reviewed runner commit. Use a
Kaggle T4 session with internet enabled. The runner masks the session to CUDA
device 0 in a fresh Python process and stops if the expected GPU or pinned
dependencies are unavailable. It reads `configs/bench.yaml` from the checkout.

## Setup and run

Set `RUN_SHA` to the full SHA of the pushed runner commit. In a Kaggle notebook,
set it with `%env RUN_SHA=<full pushed SHA>` in a Python cell and use `%%bash`
cells for the commands below. Use the environment's existing pinned packages
when available. If setup requires a different version or CUDA build, stop for
review.

```bash
set -eu
git clone https://github.com/devsutharsystems/specserve.git /kaggle/working/specserve
cd /kaggle/working/specserve
git fetch origin
git checkout --detach "$RUN_SHA"
test "$(git rev-parse HEAD)" = "$RUN_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
printf '%s  %s\n' f8d753325a1cc4e52d809e311e47f8182474c6c1504f4c29cd977d1f91289c36 bench/prompts.jsonl | sha256sum -c -
```

If needed, install the approved packages in the Kaggle environment before
launching a fresh Python process.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
python -m pip install 'transformers==5.16.1' 'tokenizers==0.23.3' datasets pytest PyYAML
```

```bash
cd /kaggle/working/specserve
python kaggle/run_baselines.py --git-sha "$RUN_SHA" \
  --output-root /kaggle/working/specserve-day2b2
```

The script verifies the frozen prompts and the pinned tokenizer and generation
config. It checks greedy against Hugging Face and prompt lookup against greedy
on the first five prompts per category, then on the full frozen benchmark. It
audits fp16 on the full benchmark before timing. Each method warms up on
`chat_qa-01` and runs three measured repeats per prompt in file order.

## Audit and resume

Read `experiments/baseline/correctness.csv` if a gate stops the run. Divergence
positions are zero-based among emitted tokens. Replay uses each method's own
decode trajectory to reconstruct its cache. The audit records the queried
shapes, cache length, shared-prefix check, both outputs' stopping decisions and
both paths' logits at the first difference. A continuation probe after an
invalid early stop is labeled in the audit.

A divergence is eligible for near-tie review only if both original-shape
replays verify the shared prefix and rank the same two divergent token IDs
first and second, with a different winner in each path. Dev judges the two
recorded logit gaps without a numeric cutoff in the runner. To record the
judgment, create a JSON object outside the checkout keyed by the audit
`case_id`. Each entry includes a `verdict` (`near_tie` or `unexplained`) and a
written `diagnosis` citing the evidence. Pass the file via
`--fp16-reviews /kaggle/working/fp16-reviews.json` on the next launch. Identical
diagnostic evidence can share a review across repeats. An unresolved untimed
divergence stops the run before timing. An eligible `near_tie` review lets
measurements proceed and is recorded in the audit as `reviewed`. Large-gap or
unexplained cases require investigation and revalidation.

Automatic speed acceptance requires all correctness gates to pass and zero
fp16 divergences across untimed audits and measured repeats, including prior
sessions of the same run. Any recorded divergence sets `requires_dev_review`
in `run.json` and keeps `speed_results_accepted` false, even with a `near_tie`
verdict. A completed run with reviewed divergences has session status
`requires_dev_review`. Dev reviews its evidence and measured results separately.

Resume by rerunning the same command against the same output root, with the
review JSON if applicable. The runner verifies the commit, prompt hash, YAML,
environment and saved repeats. It reruns both fp32 gates and the untimed fp16
audit, runs a fresh warmup for each method with outstanding repeats, and checks
saved outputs against the new fp16 reference. When the environment or commit
changes, use a fresh output root.

Each completed repeat is stored atomically under
`experiments/baseline/repeats/`. After a runtime reset, `/kaggle/working` may be
empty. Restore the full persisted `experiments/` tree beneath the same output
root before resuming. If there is no saved copy, start a fresh run.

## Export

After the run finishes or is interrupted, export the artifacts while the session
is still available. Save any review input alongside the archive.

```bash
tar --exclude='.pending-*' -czf /kaggle/working/specserve-day2b2.tar.gz \
  -C /kaggle/working/specserve-day2b2 experiments
```

Return the full tree for validation and resume. `experiments/env.txt` records
the environment, seeds and run identity. `experiments/baseline/` contains the
exact YAML, run manifest, repeat checkpoints, and the timing and iteration
CSVs for completed measurements. `correctness.csv` has the current session's
audit. `audits/` preserves each divergent session's audit, including replay
evidence, both logit gaps and any matching written review. Completed runs that
pass the gates or have reviewed divergences include `summary.json` with
per-prompt medians, min–max spread, and undefined repeat counts.
`iteration_summary.csv` contains per-repeat and pooled iteration counts and
rates. The raw iteration CSV includes the stopping iteration, which the
iteration summaries exclude.
