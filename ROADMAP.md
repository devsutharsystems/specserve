# SpecServe — 20-Day Roadmap
> Start date: 18th November 2026. Calendar dates below are outdated; follow the day numbers, not the dates.

> A small language model trained from scratch, plus a speculative decoding engine that uses it to make TinyLlama-1.1B generate faster with **identical output** — measured honestly on a free Kaggle T4.

**Schedule:** Day 1 = Fri 23 Oct 2026 → Day 20 = Wed 11 Nov 2026. ~3 hrs/day, ~60 hrs total (~8 hrs of it is learning).
**This file is the source of truth (the PRD).** Execute it; don't redesign it.

---

## Project at a glance

| | |
|---|---|
| **What** | Train a ~36M-param draft model from scratch on TinyLlama's tokenizer. Build a greedy speculative decoding engine: the draft guesses *k* tokens, TinyLlama verifies all of them in one forward pass, matching guesses are kept, the first mismatch is replaced by TinyLlama's token. |
| **Why** | First from-scratch, systems-level, independent project. Teaches LLM training end to end, KV caches, and memory-bound decoding — core ML/ML-systems interview topics. Ends in one honest number. |
| **How it works** | At batch size 1, decoding is memory-bandwidth bound: each step mostly reloads the model's weights. Verifying *k* tokens in one pass costs about the same as generating one, so every accepted guess is nearly free. |
| **How we build it** | PIV loop in Claude Code (prime → plan → review gate → execute → validate → commit → evolve), one plan per block of days below. |
| **Where** | Local CPU (Claude Code) for all code and unit tests. Kaggle T4 for training and benchmarks. Public repo: `github.com/devsutharsystems/specserve`. |
| **Resources** | Kaggle T4 (free, ~30 GPU-hrs/week), TinyLlama-1.1B-Chat weights, FineWeb-Edu sample, UltraChat 200k, W&B free tier, GitHub, Claude Code. **Cost: ₹0. No paid API.** |

---

## Locked scope

| | |
|---|---|
| **Target model** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via Hugging Face, frozen, fp16 (verify model ID and license on Day 1) |
| **Draft model** | Own code. d_model 512, 6 layers, 8 heads, RoPE, RMSNorm, SwiGLU (hidden 1408), pre-norm, tied embeddings, context 1024, TinyLlama tokenizer (32K vocab). ≈36M params by arithmetic — the Day 3 param counter confirms or corrects |
| **Decoding** | Greedy speculative decoding, batch size 1 |
| **Drafts compared** | Draft-A: pretrained from scratch on general text. Draft-B: Draft-A + distillation from TinyLlama's predictions |
| **Baselines** | Plain greedy TinyLlama; prompt-lookup decoding (drafts by copying n-grams from the prompt, no model) |
| **Hardware** | Kaggle T4, 16 GB. fp16 + GradScaler (T4 has no native bf16) |

## Research questions

1. **RQ1 — Draft quality:** does distilling the draft on the target (Draft-B) raise acceptance rate and wall-clock speedup over plain pretraining (Draft-A)?
2. **RQ2 — Where it breaks:** what *k* is best on a T4, and under which prompt types is speculative decoding *slower* than plain decoding? Does the theoretical speedup formula predict the measured one?
3. **RQ3 — Is the model worth it:** does a trained 36M draft beat model-free prompt-lookup decoding, and on which task categories?

---

## Hard rules

1. **Real, measured numbers only.** No number appears in the README unless a file in `experiments/` produced it. No placeholder or "expected" numbers anywhere.
2. **Benchmark set is frozen on Day 2** and never used for training, distillation, or tuning.
3. **Correctness before speed.** fp32 exact-match vs plain greedy must be 100% before any speed benchmark counts.
4. **Timing protocol:** batch 1, warmup runs, `torch.cuda.synchronize()` around timed regions, 3 repeats, report the median and the spread.
5. **Never skip a gate.** A failed gate shifts the schedule; it doesn't get waved through.
6. **No implementation without an approved `plans/` file.**

---

## Repo skeleton

```
specserve/
  model/        config.py, layers.py (RMSNorm, RoPE, SwiGLU, attention), draft.py, kv_cache.py
  train/        data.py, pretrain.py, distill.py
  decode/       greedy.py, prompt_lookup.py, speculative.py
bench/          prompts.jsonl (frozen Day 2), run_bench.py, analyze.py
configs/        draft_a.yaml, distill.yaml, bench.yaml
kaggle/         notebooks/scripts for every GPU run
tests/          unit + correctness tests (all runnable on CPU)
experiments/    CSVs, logs, plots — every reported number lives here
docs/           project_spec.md, research_questions.md, compute_budget.md, analysis.md
plans/          one PIV plan per block
```

---

## Week 1 — Foundations and the draft model (Days 1–6)

### Day 1 — Orientation and scaffold · Fri 23 Oct · Sonnet, medium
- **Learn:** Karpathy, *Let's build GPT: from scratch* (~2 hrs). Focus on attention, the causal mask, and the training loop.
- **Build:** repo skeleton, git init, `docs/project_spec.md`, `docs/research_questions.md`, `configs/` stubs.
- **Validate:** spec and research questions written; repo pushed.
- **Deliverable:** scaffolded repo, first commit.

### Day 2 — Theory, frozen benchmark, baselines · Sat 24 Oct · Sonnet, medium
- **Learn:** *Transformer Inference Arithmetic* (why decoding is memory-bound); Leviathan et al. sections 1–3. Derive on paper: expected tokens per target pass = (1 − α^(k+1)) / (1 − α).
- **Build:**
  - `bench/prompts.jsonl`: ~100 prompts in 4 categories — chat Q&A (UltraChat **test** split only), summarization/extraction (copy-heavy), code, open-ended writing. `max_new_tokens` 128, prompt + output ≤ 1024 tokens. **Freeze and commit.**
  - Kaggle: plain greedy TinyLlama and prompt-lookup baselines on the frozen set; record tokens/sec and environment versions (GPU, CUDA, torch, transformers).
- **Validate:** timing protocol followed; 3 repeats agree within a spread you record.
- **Deliverable:** `experiments/baseline/*.csv`, `experiments/env.txt`.
- **Mechanism question:** why does verifying 5 tokens cost about the same as generating 1?

### Day 3 — Draft model architecture · Sun 25 Oct · **Opus, high**
- **Learn:** core equations of RoPE, RMSNorm, SwiGLU.
- **Build:** config dataclass; RMSNorm; RoPE; causal attention with `F.scaled_dot_product_attention`; SwiGLU; pre-norm block; model with tied embeddings; init normal(0, 0.02) with residual output projections scaled by 1/√(2·n_layer); param counter by component; TinyLlama tokenizer loading.
- **Validate:** output shape (B, T, 32000); param count matches ≈36M or the difference is explained; initial loss ≈ ln(32000) ≈ 10.37; RoPE test (q·k depends only on relative offset); RMSNorm matches its formula.
- **Deliverable:** `specserve/model/`, passing tests.
- **Mechanism question:** how does RoPE encode relative position without a position table?

### Day 4 — KV cache and correctness tests · Mon 26 Oct · **Opus, high**
- **Build:** KV cache in the draft (append, `crop(n)`); incremental generation path.
- **Validate (all fp32, CPU):**
  1. Causal perturbation test: changing token *t* never changes logits at positions < *t*.
  2. Cached generation == uncached generation, token for token.
  3. Crop-then-regenerate == fresh generation.
  4. No NaNs; every parameter receives a gradient.
- **Known silent bug:** with a cache and more than one query token, `is_causal=True` in SDPA aligns the mask to the top-left, which is wrong. Build an explicit bottom-right-aligned mask for that case (verify against the PyTorch docs).
- **Deliverable:** `kv_cache.py`, tests.
- **Mechanism question:** after a rejected guess, what goes wrong if the cache isn't cropped?

### Day 5 — Overfit gate · Tue 27 Oct · **Opus, high**
- **Build:** overfit ~100 sequences of 256 tokens on CPU with a minimal training step.
- **Validate:** loss falls from ≈10.4 to < 0.1. **HARD GATE — no Day 6 until this passes.**
- **Deliverable:** `tests/test_overfit.py`, loss log.
- **Mechanism question:** why does a falling loss *not* prove the causal mask is correct, and which test does?

### Day 6 — Training infrastructure and Kaggle smoke run · Wed 28 Oct · Sonnet, medium
- **Learn:** training sections of Karpathy's *Let's reproduce GPT-2* (mixed precision, gradient accumulation, LR schedule).
- **Build:**
  - Data: stream a FineWeb-Edu sample, tokenize with the TinyLlama tokenizer (EOS between documents), uint16 memmap, train/val split.
  - Loop: AdamW (β 0.9/0.95, weight decay 0.1 on 2D params only), warmup + cosine, gradient accumulation, fp16 autocast + GradScaler, grad clip 1.0, W&B logging.
  - Checkpoint every N steps to `/kaggle/working`: model, optimizer, scaler, scheduler, RNG states, step, data position. Resume from it.
- **Validate:** CPU resume test — 100 steps straight vs 50 + kill + resume + 50 gives identical losses. Kaggle smoke run measures real tokens/sec and peak memory.
- **Deliverable:** `docs/compute_budget.md` — Draft-A token budget sized to ~5–6 GPU-hours **from the measured throughput**, plus the full GPU-hour plan.

---

## Week 2 — Training the drafts and building the engine (Days 7–12)

### Day 7 — Launch Draft-A; toy speculative loop · Thu 29 Oct · Sonnet, medium
- **Build (Kaggle):** launch Draft-A pretraining with the Day 6 budget.
- **Build (local, while it trains):** speculative loop v0 (greedy) using two tiny random models of your draft class, CPU fp32.
- **Validate:** draft == target → 100% acceptance and output == greedy; random draft → output still == greedy; exact for every k in 1..8.
- **Deliverable:** `decode/speculative.py` v0, toy tests.

### Day 8 — Finish and evaluate Draft-A · Fri 30 Oct · Sonnet, medium
- **Build:** finish/resume training; val loss on FineWeb val and on chat-format text; sample generations; loss-curve plot. Edge-case tests for the loop: first guess rejected; all k accepted (bonus token); EOS inside an accepted block; `max_new_tokens` reached mid-block.
- **Validate:** smooth loss curve with no divergence; checkpoint loads locally on CPU; edge-case tests pass.
- **Deliverable:** `experiments/draft_a/` (loss curve, metrics CSV, checkpoint reference).

### Day 9 — Real engine with HF TinyLlama · Sat 31 Oct · **Opus, high**
- **Build:** target = HF TinyLlama with `past_key_values`; one verification forward over k draft tokens; crop the target cache (`DynamicCache.crop` if your transformers version has it — verify; otherwise slice tensors); crop the draft cache; BOS and chat-template handling.
- **Validate:** **fp32 exact match vs plain HF greedy = 100%** on 20 prompts for k ∈ {1, 2, 4, 8}.
- **Deliverable:** working engine, `tests/test_exact_match.py`.
- **Mechanism question:** walk through one iteration where guess 3 of 5 is rejected — how many tokens are emitted, and what state is each cache in afterwards?

### Day 10 — fp16 correctness audit and first real number · Sun 1 Nov · **Opus, high**
- **Build:** fp16 mode on the T4. For every fp16 mismatch vs plain fp16 greedy, log the top-2 logit gap at the divergence point. First speed run: Draft-A, k = 4, on a subset of prompts.
- **Know:** a multi-token verification pass and a one-token pass can produce slightly different fp16 logits, so near-ties can flip. That is numerics, not a bug. fp32 stays the correctness reference.
- **Validate:** every fp16 mismatch is a near-tie. Any mismatch with a large gap is a bug — stop and fix.
- **Deliverable:** first measured speedup (whatever it is) in `experiments/`.

### Day 11 — Distillation: Draft-B · Mon 2 Nov · Sonnet, high
- **Learn:** knowledge distillation — KL(p_target ‖ p_draft), temperature.
- **Build:** `train/distill.py`. Initialize from Draft-A. Data: UltraChat 200k **train** split in TinyLlama's chat template (verify the template). Teacher runs fp16 with no grad. Loss = KL on all positions (optionally mixed with cross-entropy). Token budget from measured throughput.
- **Validate:** KL = 0 when draft logits equal teacher logits; loss decreases; leakage check — no benchmark prompt appears in distillation data (exact and near-duplicate).
- **Deliverable:** distillation launched on Kaggle.

### Day 12 — Finish Draft-B; offline acceptance eval · Tue 3 Nov · Sonnet, medium
- **Build:** finish distillation. On the frozen benchmark, measure acceptance rate α and mean tokens per target pass for Draft-A vs Draft-B (no timing needed).
- **Validate:** compare measured tokens/pass with the formula from Day 2. The formula assumes independent acceptances; any gap is a finding, not an error to hide.
- **Deliverable:** `experiments/acceptance/*.csv` — first answer to RQ1.

---

## Week 3 — Benchmark, analyze, ship (Days 13–20)

### Day 13 — Benchmark harness · Wed 4 Nov · Sonnet, medium
- **Build:** `bench/run_bench.py` over the grid: method ∈ {greedy, prompt-lookup, spec-A, spec-B} × k ∈ {1, 2, 3, 4, 6, 8} × 4 categories. Record decode tokens/sec, end-to-end latency, α, tokens/pass, peak memory, exact-match flag, environment info.
- **Validate:** the same config run twice agrees within a measured spread; the greedy baseline reproduces the Day 2 number.

### Day 14 — Full benchmark run · Thu 5 Nov · Sonnet, medium
- **Build:** run the full grid on Kaggle. Plots: speedup vs k per draft; α per category; speedup per category; every config where speculative is slower.
- **Deliverable:** `experiments/bench/results.csv` + plots — answers to RQ2 and RQ3.

### Day 15 — Failure analysis and profiling · Fri 6 Nov · Sonnet, high
- **Build:** time one speculative iteration by part: draft steps, target verification, cache crops, Python overhead. Measure the cost ratio c (draft step time / target step time) and plug it into the theoretical speedup formula; compare predicted vs measured.
- **Deliverable:** `docs/analysis.md` with all findings, including negative ones.
- **Mechanism question:** why does a larger k eventually make things slower?

### Days 16–17 — Stretch or buffer · Sat 7 – Sun 8 Nov
- **Only if every gate through Day 15 passed on schedule.** Otherwise this is catch-up time.
- **Recommended stretch:** stochastic speculative sampling (rejection sampling, Chen et al.) with a statistical test that its output distribution matches plain sampling. It completes the algorithm interviewers ask about.

### Day 18 — README draft · Mon 9 Nov
- Problem, how it works, architecture diagram (only what was built), draft training, correctness evidence, benchmark results, analysis, limitations (single seed, one GPU type, batch size 1, small draft budget), reproduction steps, one line of future work.

### Day 19 — Clean and reproduce · Tue 10 Nov
- Clean clone → CPU test suite passes; every README number traced to a file in `experiments/`; repo tidy.

### Day 20 — Ship · Wed 11 Nov
- Buffer for anything that slipped; tag `v1.0`; write the resume line with measured numbers; write answers to the 10 hardest interview questions in `docs/interview_prep.md`.

---

## Tripwires and fallback

| When | If | Then |
|---|---|---|
| Day 5 | Overfit gate fails | Fix it. Everything shifts; nothing proceeds on a failing gate |
| Day 8 | Draft-A training not finished | Use the latest checkpoint; report it as undertrained |
| Day 9–10 | fp32 exact match below 100% | Stop all benchmarking until fixed |
| Day 12 | Behind schedule | Drop Draft-B |
| Any time | Speculative decoding is slower everywhere | That is the result. Explain why in the analysis; don't tune the benchmark to hide it |

**Cut order:** stretch → Draft-B → k-grid down to {2, 4, 8} → categories down to 2.
**Never cut:** Draft-A trained from scratch, the engine, fp32 exact-match tests, greedy and prompt-lookup baselines, honest analysis.

## Out of scope

Tree drafting (Medusa/EAGLE), batch size > 1 or continuous batching, custom CUDA/Triton kernels, quantization, serving API or UI, other target models. One line of future work in the README is allowed; building any of them is not.

## Model and effort settings

| Days | Setting |
|---|---|
| 3, 4, 5, 9, 10 | **Opus, high** — silent-bug days |
| 11, 15 | Sonnet, high |
| All others | Sonnet, medium |

A stronger model doesn't replace the gates. The overfit test and the exact-match tests catch the silent bugs.

## PIV plan blocks

`plans/day01-02.md` · `day03-05.md` · `day06-08.md` · `day09-10.md` · `day11-12.md` · `day13-15.md` · `day16-17.md` · `day18-20.md`
Each plan states goal, files, tasks, Kaggle needs, and a validation strategy written before any code. No execution before review approval.

## GPU-hour estimate (replace with Day 6 measurements)

| Item | Estimate |
|---|---|
| Baselines + smoke runs | ~1 hr |
| Draft-A pretraining | ~5–6 hrs |
| Draft-B distillation | ~3–4 hrs |
| Correctness, acceptance, benchmarks | ~4–5 hrs |
| Stretch | ~1–2 hrs |
| **Total** | **~15–18 hrs over 3 weeks** (Kaggle allows ~30/week) |

## Final resume line (fill only with measured numbers)

> Trained a [N]M-parameter draft language model from scratch and built a speculative decoding engine that speeds up TinyLlama-1.1B by [X]x on a T4 with token-identical greedy output; distillation raised draft acceptance from [A]% to [B]%, and benchmarks across [C] task types identify where speculation stops paying off.

## References (verify before citing)

- Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding* — https://arxiv.org/abs/2211.17192
- Chen et al. — *Accelerating LLM Decoding with Speculative Sampling* — https://arxiv.org/abs/2302.01318
- Zhou et al. — *DistillSpec* — https://arxiv.org/abs/2310.08461
- Medusa — https://arxiv.org/abs/2401.10774 · EAGLE — https://arxiv.org/abs/2401.15077
- Su et al. — RoPE — https://arxiv.org/abs/2104.09864 · Zhang & Sennrich — RMSNorm — https://arxiv.org/abs/1910.07467 · Shazeer — SwiGLU — https://arxiv.org/abs/2002.05202
- TinyLlama — https://github.com/jzhang38/TinyLlama
- Prompt-lookup decoding — https://github.com/apoorvumang/prompt-lookup-decoding
- Hugging Face assisted generation — https://huggingface.co/blog/assisted-generation
- Transformer Inference Arithmetic — https://kipp.ly/transformer-inference-arithmetic/
- Karpathy — *Let's build GPT* — https://www.youtube.com/watch?v=kCc8FmEb1nY · *Let's reproduce GPT-2* — https://www.youtube.com/watch?v=l8pRSuU81PU · nanoGPT — https://github.com/karpathy/nanoGPT
