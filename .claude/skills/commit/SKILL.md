---
name: commit
description: Commit one day's finished work as "Day N: <summary>", staging only this project's paths and refusing fabricated or placeholder results. User-invoked only.
argument-hint: Day N [summary]
disable-model-invocation: true
---

# Commit

Input: `$ARGUMENTS` (e.g. "Day 5" or "Day 5: trained BPE tokenizer, added token-efficiency analysis").

This project is its own standalone git repo. First run `git rev-parse --show-toplevel`; if it does not print the SpecServe project folder, stop: you are in the wrong repo (e.g. a parent repo with unrelated changes) and must not commit.

## Steps

1. Run `git status --short` and `git diff --stat`. Summarize what changed for the user.
2. **Result integrity check** before staging. Refuse to commit (and say why) if any changed file contains:
   - numbers presented as results that no log/CSV/test output in `experiments/` supports;
   - placeholder values (`TODO`, `TBD`, `XX`, `[X]`, lorem-style filler, dummy metrics) in results tables, READMEs, or analysis docs;
   - result artifacts from a Kaggle run that has not actually happened;
   - secrets (API keys, `.env`, tokens), checkpoints, or large data files that should be gitignored.
   Pending-by-design markers like "pending Kaggle run" in plans are fine.
3. Check the block's success criteria in its plan file in `plans/` (e.g. `plans/day03-05.md`) against files on disk. If criteria are unmet, tell the user and ask whether to commit as partial work, labeled as such in the message.
4. Stage by **explicit path**, listing the files you intend to commit. Avoid `git add -A` and `git add .` so nothing unreviewed (images, data, checkpoints) slips in.
5. Message format: subject `Day N: <summary>` (imperative, specific), optional short body listing deliverables, then the attribution trailer required by the session's system reminder (`Co-Authored-By: ...`).
6. Commit, then show `git log -1 --stat`. Do not push unless asked.

Never use `--no-verify` or amend earlier commits unless asked.
