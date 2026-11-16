---
name: evolve
description: Given a bug, wrong assumption, or surprise from a day's work, propose a minimal edit to CLAUDE.md or a .claude/reference/ doc so it does not recur. Use after a failure or unexpected result.
argument-hint: <bug or surprise>
---

# Evolve

Input: `$ARGUMENTS`, a description of what went wrong or surprised us (or, if empty, the most recent problem in the conversation).

## Steps

1. Identify the root cause in one sentence: missing rule, wrong assumption, unclear instruction, or genuinely one-off. If one-off, say so and propose nothing.
2. Decide the right home:
   - A rule needed every session and short → `CLAUDE.md` (keep it under ~200 lines; check with `wc -l`).
   - Longer or situational guidance → a file in `.claude/reference/`, with a one-line pointer in CLAUDE.md only if not already covered.
   - A workflow-step problem → `docs/WORKFLOW.md` or the relevant skill.
3. Check the target for an existing rule that already covers it; prefer sharpening it over adding a new one.
4. **Propose the smallest edit** as a diff (old → new text) with a one-line rationale. Do not apply it until the user approves.
5. On approval, apply it and confirm CLAUDE.md is still under ~200 lines.

Do not add rules that restate ROADMAP.md, and never loosen the hard rules (real numbers only, day dependencies, out-of-scope list, Kaggle-only GPU training).
