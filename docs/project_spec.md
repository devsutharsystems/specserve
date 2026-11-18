# Project Spec — SpecServe

Status: v0, written before any code. Source of truth for scope is ROADMAP.md.

## What it is
SpecServe trains a ~36M-parameter draft language model from scratch on TinyLlama's tokenizer,
then uses it in a greedy speculative decoding engine to speed up TinyLlama-1.1B-Chat.
The draft guesses k tokens; TinyLlama checks all k in one forward pass; the longest matching
prefix is kept, and the first mismatch is replaced by TinyLlama's own token. If all k guesses
match, TinyLlama's verification pass also yields one bonus token for free. Both KV caches are
cropped to the accepted length. Output must be token-identical to plain TinyLlama greedy decoding.

This is a reproduction and measurement study, not a new method. Speculative decoding
(Leviathan et al. 2022) and trained drafts (DistillSpec, Medusa, EAGLE) already exist.
The contribution is an end-to-end build, correctness proof, and honest measurements on a free T4.

## Why speculation helps (memory-bound decoding)
- At batch size 1, each decode step reloads all target weights from GPU memory to produce one token.
  The arithmetic per weight is tiny, so step time is set by bytes moved, not FLOPs.
- A pass over k+1 tokens reuses each loaded weight k+1 times. Arithmetic intensity is about
  (k+1) FLOPs per byte of fp16 weights, versus a T4 ridge point of roughly 200 FLOPs/byte
  (derived: 65 TFLOPS / 320 GB/s). So for small k, verification should cost about the same as
  one step. This is a hypothesis to measure (Day 2, Day 15), not an assumption.
- Expected tokens per target pass: (1 - a^(k+1)) / (1 - a), where a = acceptance rate
  (assumes independent acceptances; deviations are findings).
- Wall-clock speedup: (1 - a^(k+1)) / ((1 - a)(c*k + 1)), where c = draft step time / target
  step time. c is MEASURED, never assumed. A speedup exists for some k only if a > c.
- Failure modes to measure: low acceptance, high c (eager PyTorch launch overhead can make the
  draft cost track its layer count, not its parameter count), verification not ~free,
  Python overhead and cache-crop cost.

## Locked scope
| | |
|---|---|
| Target | TinyLlama/TinyLlama-1.1B-Chat-v1.0, frozen, fp16 on GPU (fp32 for correctness tests) |
| Draft | Own code. d_model 512, 6 layers, 8 heads, RoPE, RMSNorm, SwiGLU (hidden 1408), pre-norm, tied embeddings, context 1024, TinyLlama tokenizer (32K vocab) |
| Draft-A | Pretrained from scratch on a FineWeb-Edu sample |
| Draft-B | Draft-A + KL distillation from TinyLlama on UltraChat train split |
| Baselines | Plain greedy TinyLlama; prompt-lookup decoding |
| Decoding | Greedy, batch size 1 |
| Hardware | Kaggle T4 (GPU), local Mac CPU (code, tests). Budget: zero, no paid APIs |
| Environment | Python 3.12.16 venv locally (checked 18 Nov 2026); match Kaggle's Python (verify Day 1) |

See docs/research_questions.md.

## Verified facts
Legend: VERIFIED = checked against a primary source on the date shown. DERIVED = arithmetic from
other rows; recheck with code. TO VERIFY = not yet checked.

| # | Fact | Status | Source / when to confirm |
|---|---|---|---|
| 1 | Expected tokens per target pass = (1 - a^(g+1)) / (1 - a) (g = k) | VERIFIED | Leviathan et al., Eq. 1 / Sec. 3.3 — https://arxiv.org/abs/2211.17192 |
| 2 | Walltime improvement = (1 - a^(g+1)) / ((1 - a)(g*c + 1)); improvement possible only if a > c | VERIFIED | Same paper, Theorem 3.8, Corollary 3.9 — https://arxiv.org/pdf/2211.17192 |
| 3 | T4: 16 GB GDDR6, peak memory bandwidth up to 320 GB/s (peak, not achieved); mixed-precision FP16 peak 65 TFLOPS | VERIFIED | NVIDIA T4 product brief — https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-product-brief.pdf ; NVIDIA T4 page — https://www.nvidia.com/en-in/data-center/tesla-t4/ |
| 4 | Draft params = 35,658,240 (embeddings 16,384,000 + 6 x 3,212,288 per layer + 512 final norm) | DERIVED | Recheck with Day 3 param counter |
| 5 | Draft is ~3% of target's parameters (35.66M / ~1.1B) | DERIVED | Recheck from loaded model, Day 2 |
| 6 | Target fp16 weights ~2.2 GB; bandwidth-only ceiling ~6.9 ms/token (~145 tok/s). A ceiling, not a prediction | DERIVED | Replace with measured greedy tok/s, Day 2 |
| 7 | Model ID TinyLlama/TinyLlama-1.1B-Chat-v1.0 exists on Hugging Face; license shown on the page is apache-2.0. No gating notice seen; first download confirms | VERIFIED 18 Nov 2026 | https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| 8 | Official config.json (repo head fe8a4ea1ffedaf415f4da2f062534de366a451e6): vocab_size 32000, max_position_embeddings 2048, tie_word_embeddings false (explicit). Also 22 layers, hidden 2048, 32 heads / 4 KV heads, intermediate 5632, stored dtype bfloat16 | VERIFIED 18 Nov 2026 | https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/config.json |
| 9 | PyTorch notice (posted 16 Sep 2026) states: 2.15, scheduled GA 28 Oct 2026, is the first release without Python 3.10 wheels (2.14 is the last with them); 3.11 becomes the minimum, 3.12 retained; Python 3.10 EOL 31 Oct 2026. It defers to pytorch.org/get-started/locally for the current list and does not settle Python 3.15 support | VERIFIED as the notice's statement (scheduled dates, not release confirmation) | https://dev-discuss.pytorch.org/t/notice-python-3-10-support-is-being-removed-from-pytorch-2-15-2-14-is-the-last-release-with-3-10-wheels/3440 |
| 10 | Kaggle (18 Nov 2026): GPU quota shown as 30h available of 30h (reset period not shown; verify). Test session had 2x Tesla T4 (15,360 MiB each), driver 580.178.04, Python 3.13.15, torch 2.11.0+cu128 (CUDA 12.8), transformers 5.16.1; internet on. Image versions can change, so every run logs its own versions | VERIFIED 18 Nov 2026 | Account quota page and a Kaggle notebook |
| 11 | HuggingFaceH4/ultrachat_200k splits: train_sft (208k), test_sft (23.1k), train_gen (256k), test_gen (28.3k); license MIT. Roadmap's "train/test" = train_sft/test_sft (decision, Day 2) | VERIFIED 18 Nov 2026 | Dataset page on Hugging Face |
| 12 | tokenizer.vocab_size == 32000 verified on Kaggle. Still TO VERIFY: len(tokenizer), the chat template in tokenizer_config.json, and DynamicCache.crop in transformers 5.16.1 | PARTLY VERIFIED | Day 3 / Day 9 |

## Success criteria
No speedup threshold is a success criterion. A speedup of 1x or below is a valid result if
measured and explained. Success is:
1. Draft-A trained from scratch by my own code; params counted by code; loss curve and checkpoint saved in experiments/.
2. Every gate passed in order: Day 4 tests (causal perturbation, cached == uncached, crop-then-regenerate), Day 5 overfit loss from ~10.4 to < 0.1.
3. fp32 exact match vs plain greedy = 100%: 20 prompts x k in {1,2,4,8} on Day 9, then the full frozen benchmark, before any speed number counts.
4. Every fp16 mismatch logged with its top-2 logit gap; all are near-ties; zero large-gap mismatches.
5. Benchmark grid run under the timing protocol (batch 1, warmup, synchronize, 3 repeats, median and spread, environment logged).
6. Each research question in docs/research_questions.md answered from files in experiments/, including negative results, plus measured c plugged into the formula and compared with measured speedup.
7. Every README number traces to a file in experiments/; a clean clone passes the CPU test suite; release tagged v1.0.
8. I can explain every mechanism and line of code in my own words (docs/interview_prep.md).

## Out of scope
Tree drafting (Medusa/EAGLE), batch size > 1 or continuous batching, custom CUDA/Triton kernels,
quantization, serving API or UI, other target models. Stretch only if every gate through Day 15
passed: stochastic speculative sampling. One line of future work in the README is allowed.