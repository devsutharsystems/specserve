---
name: execute
description: Implement an approved block plan from plans/ (e.g. plans/day03-05.md), running only local CPU checks. Writes Kaggle-ready scripts for GPU work and stops. User-invoked only.
argument-hint: plans/dayNN-MM.md
disable-model-invocation: true
---

# Execute

Input: `$ARGUMENTS`, a path to an approved plan such as `plans/day06-08.md`.

## Steps

1. Read the plan, `CLAUDE.md`, and the plan's ROADMAP day. If the plan path is missing or the plan was never reviewed by the user, stop and say so.
2. Re-check dependencies from files on disk. If an earlier gate is unmet, stop.
3. Implement the plan's tasks in order, matching existing code style. Follow CLAUDE.md conventions: YAML-config-driven runs, device chosen by availability, seeded, no secrets in code.
4. **Run only local CPU checks** from the plan's validation strategy (unit tests, shape/mask/NaN checks, tiny-batch smoke runs, the overfit test where the plan says so). Do not start any GPU or long training run.
5. **GPU work:** write a Kaggle-ready script/notebook cells per `.claude/reference/environment.md`, state exactly what the user must run and what to bring back, then **stop**. Do not write results for runs that have not happened.
6. **Confirm each deliverable exists as a file.** Walk the plan's success criteria; for each, run `ls`/read the file or show the test output. Mark any that depend on a Kaggle run as *pending*, not done.
7. Update `.claude/reference/day-checklist.md` only for criteria actually verified.

## Report

- What was implemented (files).
- Local checks run, with real pass/fail output. If a check failed, say so; do not paper over it.
- Deliverables: exists / pending-Kaggle / missing.
- What the user must do next (Kaggle run, review, `/commit Day N`).

Do not commit. Real, measured numbers only: never invent a metric, loss, or score to fill a table.
