Read ROADMAP.md for current status and .claude/reference/ for the checklist and environment notes.

# CLAUDE.md — SpecServe

## Project

SpecServe: a ~36M-parameter draft language model trained from scratch, plus a greedy speculative decoding engine that uses it to make TinyLlama-1.1B-Chat generate faster with token-identical output, benchmarked honestly on a Kaggle T4.

**`ROADMAP.md` is the source of truth** for scope, sequencing, gates and deliverables. Execute it. Do not redesign it, reorder it, or add scope.

## Who you're working with

- Dev Suthar, 4th-year B.E. Computer Engineering student (GTU). Background is applied ML: classical ML models, dashboards, cloud deployment, LLM-API apps.
- Has **not** previously written a transformer, a KV cache, or an LLM training loop from scratch. This project is where he learns them.
- Never attribute any other background, institution, or prior project to him — not in docs, README, commit messages, or resume text.
- Goal: a project he can defend line by line in ML / ML-systems interviews. **Understanding beats speed.**

## Your role

You implement inside this repo: code, tests, configs, Kaggle scripts, git. Plans are reviewed outside this session before execution.

- **Teach while building.** Before implementing a new mechanism (RoPE, KV cache, distillation loss, the speculative loop), explain it in a few plain sentences. Keep code readable and commented where the math is non-obvious.
- Ask at most one clarifying question at a time.
- If something in ROADMAP.md looks wrong, flag it and wait — don't silently "fix" the plan.

## Workflow (PIV loop) — mandatory

1. **Prime:** read CLAUDE.md, ROADMAP.md, `git log --oneline -20`, and `plans/`.
2. **Plan:** write `plans/dayNN-MM.md` for the current block: goal and success criteria, files to create/change, task list, Kaggle needs, and a **validation strategy written before any code**.
3. **Review gate:** stop after writing the plan. **No implementation without an approved plan file.** If no approved plan exists for the current block, do planning only.
4. **Execute:** implement the approved plan. Run CPU tests locally. Anything needing a GPU becomes a Kaggle script/notebook for Dev to run — never assume you can run GPU jobs.
5. **Validate:** run the plan's validation strategy and report results exactly as measured.
6. **Commit:** `Day N: <what was done>` (e.g. `Day 4: draft KV cache + causal perturbation and cache-equivalence tests`).
7. **Evolve:** after any bug or surprise, propose one concrete addition to this file or a test so it can't recur.

**Sequencing is enforced.** Never skip a gate (Day 5 overfit test, Day 9–10 exact-match). If asked to jump ahead past an unmet gate, say which gate blocks it instead of complying.

## Non-negotiable rules

1. **Real, measured numbers only.** Never invent, estimate, or round up results. Never write placeholder numbers into README, docs, or plots. If something fails, report the failure.
2. Every reported number must be produced by code and saved in `experiments/` (CSV, log, or plot) with the config and environment that produced it.
3. The benchmark set `bench/prompts.jsonl` is frozen once committed (Day 2). Never train, distill, or tune on it. Distillation uses the UltraChat **train** split; the benchmark uses the **test** split.
4. **Correctness before speed:** no speed result counts until fp32 exact-match vs plain greedy is 100%.
5. Stay inside scope. Out of scope (do not build, do not suggest "while we're at it"): tree drafting (Medusa/EAGLE), batch size > 1 or continuous batching, custom CUDA/Triton kernels, quantization, serving API/UI, other target models. One line of future work in the final README is allowed.

## Environment

- **Local:** CPU only. Used for all code, unit tests, toy-model tests, and the overfit gate. Tests must run on CPU in under a few minutes.
- **Kaggle:** T4 16 GB. Used for baselines, training, distillation, benchmarks. Sessions die — every long job must checkpoint to `/kaggle/working` and resume.
- **No paid APIs.** Budget is zero.

## Stack

Python 3.10+, PyTorch, Hugging Face `transformers` (target model + tokenizer only), `datasets` (FineWeb-Edu sample, UltraChat 200k), PyYAML configs, pytest, W&B free tier, pandas + matplotlib for results.

The draft model is **our own code** — never `nn.Transformer`, never an HF model class. HF is used only for the frozen target and the tokenizer.

## Repo layout

```
specserve/model/    config.py, layers.py, draft.py, kv_cache.py
specserve/train/    data.py, pretrain.py, distill.py
specserve/decode/   greedy.py, prompt_lookup.py, speculative.py
bench/              prompts.jsonl, run_bench.py, analyze.py
configs/            *.yaml — all hyperparameters live here, not in code
kaggle/             one script/notebook per GPU run
tests/              CPU-runnable unit and correctness tests
experiments/        all results (CSV, logs, plots, env info)
docs/               spec, research questions, compute budget, analysis
plans/              PIV plans
```

## Engineering conventions

- **Seeds everywhere** (Python, NumPy, torch, data order). Same seed and settings across compared runs; change one thing per experiment.
- **Configs drive runs.** Each experiment's YAML is saved alongside its results.
- **Checkpoints** contain model, optimizer, GradScaler, scheduler, all RNG states, step, and data position. Resume must reproduce an uninterrupted run exactly on CPU.
- **Timing protocol** for any speed number: batch size 1, warmup runs, `torch.cuda.synchronize()` around timed regions, 3 repeats, report median and spread, log GPU/CUDA/torch/transformers versions.
- Mark anything uncertain about an external API or model ID as "verify" and check it before relying on it.

## Known traps (read before touching these areas)

- **SDPA causal mask with a cache:** `F.scaled_dot_product_attention(is_causal=True)` aligns the mask top-left when query length ≠ key length. With a KV cache and more than one query token, build an explicit bottom-right-aligned mask. This bug produces fluent but wrong output.
- **A falling loss does not prove the mask is correct.** The causal perturbation test does: changing token *t* must not change logits at positions < *t*.
- **Cache crop after rejection:** both the draft and target caches must be cropped to the accepted length every iteration. Off-by-one accept counts and stale caches are the most common speculative-decoding bugs.
- **fp16 near-ties:** a multi-token verification pass and a single-token pass can produce slightly different fp16 logits, flipping near-tied argmaxes. fp32 is the correctness reference; every fp16 mismatch must be logged with its top-2 logit gap. A large-gap mismatch is a real bug.
- **T4 has no native bf16.** Use fp16 autocast + GradScaler.
- **Token storage:** TinyLlama's vocab is 32,000, so uint16 memmaps are safe.
- **Target cache cropping:** check whether the installed `transformers` version provides `DynamicCache.crop`; if not, slice the cached tensors directly.
- **Expected initial loss** for a fresh draft ≈ ln(32000) ≈ 10.37. A very different value means an init or loss bug.

## Model / effort reminder

Opus, high effort on Days 3, 4, 5, 9, 10 (silent-bug days). Sonnet, high on Days 11 and 15. Sonnet, medium otherwise. A stronger model does not replace the gates.

## Definition of done (whole project)

Draft-A trained from scratch; engine passes fp32 exact-match on the frozen benchmark; benchmark grid run vs greedy and prompt-lookup baselines; analysis explains where speculation helps and where it loses; README contains only numbers traceable to `experiments/`; a clean clone passes the CPU test suite; release tagged `v1.0`.
