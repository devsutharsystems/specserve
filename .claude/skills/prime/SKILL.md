---
name: prime
description: Load SpecServe project context at the start of a session and report the current roadmap day, block, and what is next. Use at the start of a session or when asked "where are we".
---

# Prime

Load context, then report. Write no code and change no files.

1. Read `CLAUDE.md` and `ROADMAP.md` in the project root.
2. Run `git log -15 --oneline` (this project is its own standalone repo; if `git rev-parse --show-toplevel` is not the SpecServe folder, say so).
3. List `plans/` and read the most recent plan file (e.g. `plans/day03-05.md`) if any. Also read `.claude/reference/day-checklist.md`.
4. Determine the current day: the earliest roadmap day whose deliverable file does not exist or whose commit is missing. Verify against files on disk, not just commit messages.

## Report (concise)

- **Current day and block:** Day N (title from ROADMAP.md), and its plan block (`day01-02`, `day03-05`, `day06-08`, `day09-10`, `day11-12`, `day13-15`, `day16-17`, `day18-20`).
- **Done:** days with committed work and existing deliverables.
- **Next:** the Learn/Build/Deliverable for the current day, in two or three lines.
- **Gates:** state of Day 2 (benchmark frozen and committed), Day 5 (overfit test passes), Day 9–10 (fp32 exact match 100%). If the likely next request would skip one, say so.
- **Model/effort:** remind if the current day is an Opus-high day (3, 4, 5, 9, 10) or Sonnet-high day (11, 15).
- **Pending Kaggle work:** any GPU run written but with no results brought back.
- **Suggested next command:** usually `/plan-feature "Days N-M"` or `/execute plans/dayNN-MM.md` if an approved plan exists.

If nothing exists yet, say Day 1 is next. Never infer results that are not in files.
