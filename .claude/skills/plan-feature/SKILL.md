---
name: plan-feature
description: Write a plan for one roadmap block ("Days 1-2", "Days 3-5", ...) to plans/dayNN-MM.md. Plans only; writes no code. Use before implementing any block.
argument-hint: Days N-M
---

# Plan Feature

Input: `$ARGUMENTS` (e.g. "Days 3-5"). Produce `plans/dayNN-MM.md` (zero-padded: `day03-05.md`). Blocks are fixed by ROADMAP.md's "PIV plan blocks" list. **Write no code and no files other than the plan.**

ROADMAP.md is the PRD. Read the entries for every day in the block (and the days they depend on); do not re-derive scope or redesign.

## Steps

1. Parse the block. Read those days in `ROADMAP.md`, plus `CLAUDE.md`. Read `.claude/reference/environment.md` if any day involves Kaggle.
2. **Dependency check.** Verify from files on disk that earlier deliverables exist and gates are met: Day 2 benchmark frozen and committed before any training; Day 5 overfit test passing before Day 6; fp32 exact match 100% (Day 9–10) before any speed benchmark counts. If not met, do not write the plan; state which gate is unmet and what to do first.
3. Inspect existing code/configs the block will touch so the plan names real files.
4. Write the plan with exactly these sections:

   - **Goal** — one or two sentences per day, from ROADMAP's Build/Deliverable.
   - **Learn** — the concepts in a few plain lines, plus ROADMAP's reading list. Dev is new to transformers/KV caches/training loops; keep it teachable.
   - **Success criteria** — checkable statements, each tied to a file or test output.
   - **Files** — create/modify list with paths matching the repo layout in CLAUDE.md.
   - **Tasks** — ordered, small, implementable steps, grouped by day.
   - **Validation strategy** — three parts:
     - *Local CPU tests:* exact commands and what passing looks like.
     - *Kaggle runs:* what Dev runs, script name, config, expected duration, what to bring back. "None" if CPU-only.
     - *Expected artifacts in `experiments/`:* file names and formats. Mark any that only exist after a Kaggle run.
   - **Mechanism question(s)** — copy the ROADMAP mechanism question for each day that has one.
   - **Risks** — what could go wrong, plus the ROADMAP tripwire/cut-order step that applies.
   - **Out of scope** — anything tempting but cut (see CLAUDE.md).

5. Tell Dev the plan is ready for external review and that `/execute plans/dayNN-MM.md` comes only after approval. Do not start executing.

Never put expected numbers in the plan as if they were results. Targets may be stated only as targets.
