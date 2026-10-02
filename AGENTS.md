# AGENTS.md

Single source of truth for all working rules. No separate `INSTRUCTIONS.md` —
its content has been split by topic into `docs/invariants/` and `docs/plans/`,
per the routing below.

## Routing

```
AGENTS.md (this file — routing + meta rules)
    ↓
docs/docs/index.md (knowledge domain map)
    ↓
docs/domain/<project>/index.md (project-specific facts: architecture, chosen libs, phase status)
    ↓
docs/invariants/ (universal rules for your current stack/domain)
    ↓
specific rule doc (core.md, backend.md, frontend.md, security.md, tooling.md, processes.md)
    ↓
source code (ground truth)
    ↓
implementation
```

Project-specific facts (this repo's folder names, model choices, phase
status) belong in `docs/domain/<project>/index.md` — never in
`docs/invariants/`. Invariants are the rules that hold across every project;
domain docs are what's true about *this* one.

## Rules

All rules live in `docs/invariants/`. Pick the rule doc that matches your
current task — never load all of them.

| Task                             | Rule doc                        |
| --------------------------------- | -------------------------------- |
| Any task                          | `docs/invariants/core.md`       |
| FastAPI / backend                 | `docs/invariants/backend.md`    |
| Next.js / frontend                | `docs/invariants/frontend.md`   |
| Credentials, DB, ORM              | `docs/invariants/security.md`   |
| Package managers, Docker, paths   | `docs/invariants/tooling.md`    |
| Plans, ADRs, cadence, README      | `docs/invariants/processes.md`  |

`core.md` is always loaded: folder structure, file-size limits,
real-data-only, and the Ponytail ladder — rules that hold no matter what
you're building. Everything else loads on demand.

Open item from this merge: the tooling row used to say "Pixi." The detailed
rules that actually existed (in the old INSTRUCTIONS.md) specify `uv` for
Python and `pnpm` for Node, with no mention of Pixi. `docs/invariants/tooling.md`
below is written for uv/pnpm. Confirm which one this repo actually uses and
fix whichever side is wrong.

## First run — bootstrapping the docs tree

If `docs/invariants/` (or a specific file in it), `docs/docs/index.md`, or
`docs/domain/<project>/index.md` don't exist yet, don't wait for them to be
handed over — generate them yourself, in this same first session:

1. Create `docs/docs/index.md` and whichever `docs/invariants/*.md` files
   the project's actual stack needs (`core.md` and `processes.md` always;
   `backend.md` / `frontend.md` / `security.md` / `tooling.md` only for the
   stacks actually in play — a Java app, for instance, skips frontend.md).
2. Write a `docs/invariants/<stack>.md` for any stack this hierarchy
   doesn't already cover (Java, Android, whatever's being built) — folder
   layout, build tool, test framework, the same kind of thing backend.md
   does for FastAPI.
3. For anything you can't infer (framework choice, build tool, test
   framework, folder convention), ask — don't guess and lock it in
   silently. Propose a first draft, get it confirmed, then write it.
4. Write `docs/domain/<project>/index.md` with whatever architecture and
   naming actually gets decided.
5. Once the tree exists, everything above applies exactly as if it had been
   there before this session — this is a one-time scaffold, not a standing
   excuse to skip checking docs later.

If `docs/domain/<project>/index.md` (or an old INSTRUCTIONS.md-style file)
already exists with real content, don't regenerate over it — read it first
and ask before replacing anything.

## Never go exploring

For anything related to a repo, check the docs first — that's what they're for.

Go into source only when: (1) the docs don't answer, (2) the user explicitly
asks for implementation details, or (3) the task is to implement and the
docs say to do X — verify X in the source first.

## Working one step at a time

This shape is fixed and always active — no need to load `processes.md` just
to know it:

1. Work ONE step at a time, from the current plan doc.
2. Test it yourself before handing off. Fix obvious errors yourself — don't
   hand basic issues back to the user.
3. Note anything else you notice along the way (UI gaps, config, missing
   checks). Flag it to the user rather than silently fixing it or silently
   skipping it — keep a running list so nothing gets lost.
4. Hand off in plain language — what changed, why, what the user will
   notice — for both frontend and backend, plus the exact terminal commands
   so the user can verify it themselves.
5. Provide a short, to-the-point terminal commit command ready to copy-paste
   (e.g., `git add -A; git commit -m "chore(setup): scaffold project and dependencies"`).
   Never run git commit autonomously; always give the command for the user to execute.
6. Wait for explicit approval before starting the next step. Mark the step
   as approved (`✅`) once confirmed.
7. Log the step in that day's `.memory/tasks.md` (format in `processes.md`)
   so progress is visible without re-deriving it from chat history.
8. Maintain `.gitignore` side-by-side on EVERY step: keep datasets, weights,
   checkpoints, `.memory/`, MLflow runs, and temp artifacts cleanly excluded
   so proposed git commits remain small and clean.
9. Empirical integrity & explicit experiment storage: Whenever running
   benchmarks, sweeps, or training experiments, execute the exact number
   of epochs, samples, and trials claimed. Never assume, approximate, or
   misreport execution depth. Persist raw run metrics and artifacts
   explicitly to disk (e.g. JSON/figures) and MLflow so all results are
   tangible, verifiable, and auditable.

This is what gives a visible timetable: the numbered plan docs under
`docs/plans/` are the "what's next," and `.memory/tasks.md` is the running
"what's done" — both are cheap to skim, so where things stand is never
buried in scrollback.

## Session-level rules

- Capture fixes where they belong. When the user asks to fix/change
  something, add the rule to the invariants doc that owns that topic — not
  by dumping everything into AGENTS.md. Only routing/meta-level changes
  belong here. Note the change (what, why, how to revert) in that day's
  `.memory/tasks.md`.
- Session continuity. Once a doc is loaded this session, its rules apply
  for the rest of the session — no re-reading needed.
- Destructive actions. Always confirm before pushing, deleting, or
  rebuilding containers.
- Commits. Never run `git commit` without explicit user confirmation. Stage
  the change, propose the message (below), let the user commit.
- Co-author. Never add the AI/agent's name as co-author.

## Git

One commit per approved unit of work — a single step, or an explicitly
batched group of steps done together. Never split one approved batch into
several commits, and never fold unrelated work into one commit.

- Subject under 72 characters, self-explanatory without a body — e.g.
  `chore(setup): scaffold dependencies and feature structure`.
- Covers only what this commit's diff actually changes. Never re-lists
  earlier or already-committed work.
- No bulleted essays in the body.

The agent proposes the message; the user runs the commit.

## Skills

- Frontend UX review/polish/audit → **impeccable**.
- Visual redesign/overhaul (new look, whole-screen redesign) →
  **ui-ux-pro-max**. Don't mix these two up.
- Backend security implementation or review → **backend-security-coder**
  (vs a general security-auditor: this one writes secure code, it doesn't
  do high-level audits).
- Simplest solution, shortest working diff → **Ponytail**, active by
  default for all code writing (full ladder in `core.md`).
- If a required skill isn't installed, tell the user how to install it and
  offer to do it for them.
