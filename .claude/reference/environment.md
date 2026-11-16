# Environment: Local CPU vs Kaggle GPU

No local GPU. Training, distillation and all speed benchmarks happen on Kaggle (T4 16 GB, ~30 free GPU-hrs/week).

## Local (this machine, CPU only)

Allowed: all unit tests, the causal-perturbation and cache-equivalence tests, the Day 5 overfit test, toy-model speculative-loop tests, fp32 exact-match tests on tiny models, data-pipeline smoke runs on small slices, analysis and plotting.

Not allowed: draft pretraining or distillation at real scale; any speed number (CPU timings are never reported as results).

## Kaggle handoff

For GPU work, write a script Dev can run in a Kaggle notebook and stop. The script must:

- State exactly which repo files / datasets to attach (or `git clone` the public repo).
- Read all settings from a YAML config path; select device by availability.
- Set seeds; log to W&B if `WANDB_API_KEY` is present, else to a local CSV.
- Use fp16 autocast + GradScaler (T4 has no bf16).
- Checkpoint to `/kaggle/working/` every N steps with full state (model, optimizer, scaler, scheduler, RNG, step, data position) and resume from the latest checkpoint automatically.
- For timing: follow the CLAUDE.md timing protocol and write environment info (GPU, CUDA, torch, transformers versions) next to the results.
- Print a clear list of what to download back into `experiments/`.

Results exist only after Dev brings them back. Only then are artifacts written to `experiments/` and cited.

## Tooling

Python + PyTorch; Hugging Face `transformers` (frozen TinyLlama target + tokenizer only) and `datasets` (FineWeb-Edu sample, UltraChat 200k); PyYAML; pytest; W&B free tier; pandas + matplotlib. No paid APIs.
