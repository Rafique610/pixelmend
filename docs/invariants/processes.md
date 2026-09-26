# Processes Invariants

## Plans

- Plans live under `docs/plans/`.
- One file per task/scope (not one monolithic plan).
- Steps are numbered within each plan file: `## Step 1: …`, `## Step 2: …`.
- Each step document includes: scope, decision(s), research, recommendation,
  verification, files changed.
- Steps with real decisions use the research template from AGENTS.md
  (alternatives table, side-by-side when cheap, evidence-based recommendation).

## Working Cadence

1. Write the step's content (decision, research, recommendation) in its plan
   file first.
2. User reviews.
3. On "implement step N," build it.
4. Hand back: what changed, exact test commands, new findings for Research Notes.
5. Provide the exact, copy-pasteable terminal commit command with a concise message (<72 chars).
6. Wait for explicit approval.
7. Mark step `✅` once approved.
8. Log in `.memory/tasks.md`.

## Commits

- One commit per approved unit of work.
- Subject under 72 chars, self-explanatory (e.g., `chore(setup): scaffold project and dependencies`).
- Covers only what this commit's diff actually changes.
- Never add AI/agent as co-author.
- Provide the exact terminal command ready to copy-paste; user runs the commit.

## README Maintenance

- Update `README.md` on EVERY step before handing off.
- Every git commit must include an up-to-date README.

## Session Memory

- `.memory/<date>/tasks.md` — one file per day, sequential entries.
- Timestamps in 12-hour AM/PM with timezone + UTC.
- Task board table format with status badges.
- Update after EVERY chat turn.

## ADRs (Architecture Decision Records)

- Not required as separate files for this project — the Research Notes
  sections within plan steps serve as inline ADRs.
- If a decision is cross-cutting (affects multiple tasks), document it in
  `docs/domain/genai-assignment/index.md` instead.
