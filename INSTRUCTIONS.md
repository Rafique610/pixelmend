# INSTRUCTIONS.md

Global project working instructions provided by the user. Follow these for ALL future work, across the whole project.

## 1. Source of truth

- The source of truth for ALL work is the `docs/` folder (e.g. `docs/additional/`, `docs/plans/`).
- Work plans live in docs, whether under `additional` or explicit plans folders.
- Before starting work, read the relevant plan in docs and follow it step by step.

## 2. Folder structure (feature-based rule)

- Every feature/domain gets its OWN top-level folder with separated responsibility (e.g. `src/memories/`, `src/assistant/`, `src/ui/`).
- Never use horizontal technical layering (e.g. do NOT split into `src/capture/`, `src/storage/`, `src/retrieval/`). Instead, group all components of a feature (models, DB, services, helpers) inside that feature's directory (`src/memories/`).
- If any other feature is added, it gets its own folder — never mix responsibilities across features.

## 3. Working cadence

- Work ONE step at a time (per the plan in docs).
- After completing a step, **do your own manual testing first** — run it, fix simple issues on the fly yourself, don't hand basic/obvious errors back to the user.
- **Review each step for missing/unfinished work** — after finishing a step, also look for related fixes you noticed (UI gaps, backend issues, config, health indicators, etc.) that aren't in the plan.
- When you spot such an idea or fix, **consult/inform the user about it** before or alongside doing it — don't silently add it. Keep a running list of suggested fixes so nothing gets lost.
- **Docker DNS sometimes fails — retry at least 3 times** when a build/pull/connect fails with a DNS/network error before giving up; it usually succeeds on a retry.
- Only when you are confident it works and have fixed errors yourself, hand the **final fixed version** to the user for their test/approval.
- **Explain each step in a very user-friendly way — for BOTH frontend and backend.** Plain language, no jargon: what the actual change does, why it matters, and what the user would see/experience. Don't assume the user knows the internals.
- **ALWAYS give the user terminal commands so THEY can test the feature themselves.** Every step handoff must include a short "You can test it yourself" section: the exact commands to run (and what to check for each), referencing the project's own tools/tasks (e.g. `task migrate-fjordvik`, `task fjordvik-check-db`, `docker exec ... psql ...`, `uv run python -c ...`). Do not only rely on the agent's own testing — the user wants to verify each feature from the terminal too.
- After the user approves, rename that step's doc to `✅ <name>.md` and provide the git commit message.
- **Never add the AI/agent's name as a contributor or co-author** in commits or commit messages. Commits belong to the user — write them plainly without agent attribution.
- **One commit message per batch, not one per step.** When 2–3+ steps are done together, give a SINGLE consolidated commit message (even if the earlier per-step messages were already shared). It must clearly describe exactly what was done in that commit so anyone reading it understands the full change. Don't dump multiple separate messages for one commit.
- **Commit message format — short and to the point.** Keep commit messages concise, clear, and direct (e.g. `chore(setup): scaffold dependencies and feature structure`). Avoid long bulleted body paragraphs or essays. Keep it strictly to the point so history is clean and fast to read.
- **Commit messages cover ONLY the changes in that commit's diff.** Never re-list earlier/previous steps or things already committed — only what this commit actually changes. This keeps history clean and each commit's scope obvious (the user's projects commit after approval, so a "batch" is typically one step's worth of changes).
- After approval, wait for explicit permission before starting the next step.
- Also apply the `✅` rename retroactively when a step is approved.

## 4. Code organization & responsibility limits

- Always split responsibilities among folders and `.py` files. NEVER make a `.py` file a "god file" doing many things at once.
- If a file exceeds ~300 lines, it has too much responsibility — split it or apply a proper design pattern.
- Use design patterns where needed, but do NOT overengineer.
- When using a design pattern, inform the user which one was used and WHY.

## 4b. FastAPI structure (mandatory)

Whenever building/working on any FastAPI app in this project:
- **Folder structure:** `routers/`, `repositories/`, `schemas/` (and `config.py`, `database.py`, `main.py`). Each router file is thin — it only wires an endpoint to a repository function. `main.py` must stay as clean/lean as possible: it just creates the app, adds middleware, and includes routers. No business logic in `main.py` or in routers.
- **Clear separation of concerns:** routers → endpoints, schemas → request/response shapes + validation, repositories → DB access (ORM only). Models/settings stay in their own files. Use appropriate design patterns (repository, facade, etc.) and tell the user which pattern and why.
- **Request schemas are mandatory on FastAPI endpoints:** every endpoint that accepts a request body (or meaningful query params) MUST declare a Pydantic `BaseModel` schema. Schemas are stored separately in the `schemas/` folder as Pydantic models. NEVER write an endpoint request in a single unnamed object. (Response schemas/typed returns are also preferred where sensible.)
- **Why schemas live separately (validation AND OpenAPI testing):** putting request/response shapes in `schemas/` is not just about validation — it keeps the auto-generated Swagger/OpenAPI docs (`/docs`) clean, typed, and self-documenting so the USER and AI agents can explore and test every endpoint directly in the browser against the real API, without reading source. A schema also gives one central place to constrain values (ranges, enums, string length, defaults), so bad input is rejected with a clear 422 before it ever touches the DB. Always tell the user how to use `/docs` to test the endpoint they just built.

## 4c. Real data only (mandatory global rule)

**NEVER show unreal/fabricated/mock data to the user in UI, API responses, or dashboards.** Every number or figure rendered must come from a real source (a real DB query/aggregate, a real file/object, a real service) — never a hardcoded/guessed/invented value, even for "nice-looking" demo stats. This applies everywhere (dashboards, KPIs, panels, labels that imply an engine/process that doesn't exist).
- Real counts/aggregates are fine. Fabricated summaries (invented percentages, "resolved"/"clean"/confidence scores, summed-total metrics that don't mean anything, implied-engines) are NOT.
- If a metric genuinely cannot be measured from real data, either compute it honestly, show only the real parts, or omit it — do not invent it.
- Keep a running awareness: when a value looks computed but is actually a guess, strip it or make it real before showing. When in doubt about whether something is mock, ask the user / investigate rather than shipping a placeholder.

## 5. Security & settings

- NEVER use environment variables directly in code. Always route them through a Pydantic settings/config (e.g. `config.py` / `Settings` class).
- NEVER expose credentials in code or output, even if they are local/test credentials.
- **NEVER hardcode credentials anywhere — not in code, not in config files, not even for local development.** This includes Docker `compose.yaml`, Alembic `env.py`, Dockerfiles, scripts, etc. Always source them from the project's `.env`/`.env.local` (or use Pydantic settings backed by env vars). Docker Compose should reference `${VAR}` (and use `$${VAR}` for container runtime env in healthchecks), and Alembic env files should build URLs from a `Settings` class — never embed `user:pass@host` strings. If a credential must exist as a working default for local dev, it lives in the env file(s), not in tracked source/config.

## 5b. Database access (ORM only)

- ALWAYS write to/read from any database through an ORM. NEVER use raw SQL queries.
- Applies everywhere: Python (SQLAlchemy) and Node/Next.js (Prisma or any other ORM).
- When a design pattern determined to be needed for a feature the user asks for, DISCUSS it with the user first before implementing.

## 6. Tooling & package management

- Use the best package manager for the framework/language:
  - Python: always use `uv` (never call `python` directly for project commands).
  - Node: use `pnpm` with proper lockfiles and package.json.
- Always have a project config + lock file (e.g. `pyproject.toml` + `uv.lock`) to track installed libraries.
- Always add a `Taskfile.yml` entry so users can run things easily instead of memorizing raw commands.

## 7. Docker

- Always make Dockerfiles TWO-STAGE so images are as small as possible and build times are lower.

## 8. Dynamic instruction capture

- Whenever the user asks to fix/change something during approval or requests new behavior, capture that instruction and add it to `INSTRUCTIONS.md` automatically.

## 8b. Creating plans (when the user asks)

When the user asks for a plan, do a deep dive first:

1. Explore the whole codebase and/or whatever the user wants to build — read existing code, patterns, docs, and conventions.
2. Then produce ONE big detailed plan.
3. Break the plan into numbered STEPS (e.g. `01-...`, `02-...`) where each step is small, testable, and has dependencies.
4. Each step document should include: What / Why / How-to-implement / Verification / Files changed / Dependencies / Common issues.
5. Save plan docs under `docs/` (e.g. `docs/additional/` or a plans folder).
6. Follow the plan step by step using the working cadence in Section 3.

## 8c. README maintenance (mandatory on every step)

- **Update README.md on EVERY step:** At the completion of each step (before handing off to the user for approval), update `README.md` with newly implemented capabilities, file paths, and terminal commands.
- The user commits after each approved step, so **every git commit must naturally include an up-to-date README**. Never leave README updates to the end.
- Always keep README synchronized with the live state of the codebase (architecture, folders, commands).

## 8d. Token / pattern caching & common project rules

- If a thinking pattern, piece of code, or approach is likely to be reused again, cache/remember it so you don't regenerate it from scratch.
- Reuse prior patterns and decisions (e.g. models, settings patterns, taskfile patterns) rather than re-deriving them.
- Keep a working memory of repeated patterns across steps:
  - **Feature-Based Architecture:** Domain features live in `src/<feature>/` (e.g. `src/memories/`, `src/assistant/`, `src/ui/`).
  - **Configuration:** Always through `Settings` in `src/config.py` using `MEMORY_` prefix. Never read env vars directly.
  - **Database:** Local SQLite in `data/memories.db`, fully parameterized queries (`?`), idempotent `init_db()`.
  - **Commit Messages:** Short and to the point (e.g. `feat(memories): add SQLite storage models`). No bulleted body essays.
  - **Session Memory:** Single daily file at `.memory/<date>/tasks.md` incremented sequentially.
  - **Testing:** Always provide terminal verification commands with every step handoff so the user can test directly.
- **Cache INSTRUCTIONS.md at read time:** the FIRST time the user tells the agent to read `INSTRUCTIONS.md`, the agent MUST cache/remember its full contents for the whole session (its rules apply for all subsequent work). Do not rely on re-reading it later or on memory files — that risks losing rules and burning tokens. If the rules need to persist across sessions, that is what `.memory/` is for; but in-session, keep INSTRUCTIONS.md rules available from the moment of the first read.

  - **Strict Phase 1 Parity:** Do not assume things. Always derive decisions, parameters, and constraints directly from the Phase 1 implementations. Keep Phase 1 constraints constant (e.g., chunk size 256, overlap 30-50, FTS configurations) when porting features to Android.

## 8e. Session memory log (.memory/)

- Maintain a memory log in `.memory/` — this is shared memory for the agent AND the user.
- Structure: one folder per day, named by date, e.g. `.memory/05-september-2026/`.
- **Single daily file:** Inside the daily folder, keep **ONE single file** (`tasks.md`) for the entire day — never split into multiple files per feature.
- **Sequential incrementation:** As work progresses throughout the day, each new task, step, or feature is appended directly into the task board table below the previous rows in chronological sequence.
- Each entry must include:
  - Date formatted as human-readable words, e.g. **05 September 2026** (not `09/05/2026`).
  - Timestamps in 12-hour AM/PM format, e.g. **1:48 AM**.
- **Multi-region teams:** always include the time zone alongside the timestamp, and always also include UTC so colleagues in any country can correlate. Format: `1:48 AM (UTC+05:00 · 20:48 UTC)`. Optionally note which user/region did the work.
- **NEVER guess or fabricate timestamps.** When writing/updating a memory entry, capture the real wall-clock time (e.g. `Get-Date`) at that exact moment. If past entries weren't captured live, mark them `(~ approx.)` with known order only, and go exact from the next entry onward.
- Use a **task board table** so at a glance you can see: what was assigned, was it approved or not, and what changed. Update this table on EVERY iteration.
- Make the board **impactful and scannable** for humans: start with a one-line day verdict, a bold summary row (X done / Y approved / Z changed), a clear status badge per task (e.g. ✅ Approved / 🔄 In progress / ⚠️ Needs changes), and a task board table.
- Log which tasks the user assigned that day, how many got approved, and how many needed changes (and what changed when you made a change).
- **Update `tasks.md` after EVERY chat turn** — never skip or batch-skip it. Every session/step/decision/task board entry must be captured so the history is complete and never gaps.
- Each task-board row must carry **enough detail to reverse it later** — e.g. "added route X in `file.py`", "added component Y", "changed fetch to use Z". If the user later says "reverse that", you (or any future session) must be able to undo it from the entry alone. Include file paths, names, and the exact thing that was added/changed.
- **Also log EVERY instruction change in INSTRUCTIONS.md** — in `tasks.md` (and any follow-up daily files), keep a dedicated section listing every change to `INSTRUCTIONS.md` that day, including:
  - instructions the user explicitly asked to add/change, AND
  - any instruction/rule we ourselves added dynamically from what the user told us (capture the source comment too).
  For each change write it VERY clearly so it can be reversed later if needed:
  - **Changed in INSTRUCTIONS.md:** *which section/file + what rule or wording was added/edited* — write it as a plain, reversible sentence, e.g. "Added Section 3 rule: do own manual testing before handing a step to the user."
  - **Why/trigger:** *what the user said that caused it* (one line).
  - **To revert:** *exact action to undo it* (e.g. "Delete that bullet from Section 3; or keep the rule but soften it to X.").
- Keep it easy to understand for BOTH the AI and a human: use tables, clear spacing, no ambiguity.
- `.memory/` is git-ignored — never commit it or share it publicly. Share it only with teammates if they ask.

## 9. Ponytail (strictly active for all code writing)

- **Ponytail is strictly active for ALL code writing and refactoring:** Always write the laziest solution that actually works — simplest, shortest, most minimal.
- **Follow the Ponytail ladder:**
  1. *Does this need to exist at all?* (YAGNI — skip speculative abstractions).
  2. *Already in this codebase?* Reuse existing functions, types, and patterns.
  3. *Standard library does it?* Use standard library (`sqlite3`, `dataclasses`, `json`, `pathlib`).
  4. *Native platform feature covers it?* Use native platform capability.
  5. *Already-installed dependency solves it?* Use it. Never add new packages without explicit need.
  6. *Can it be one line?* Make it one line.
  7. *Only then:* write the minimum code that works.
- **No unrequested abstractions:** No factories for one product, no speculative interfaces, no premature generalizations.
- **Bug fixes address root causes:** One shared guard over multiple caller patches.
- **Shortest working diff wins:** Every non-trivial piece of code gets ONE minimal runnable verification check.

## 10. UI/UX skill routing

- If the user asks anything **UX-related** (review, improve, critique, polish, audit a specific interface / component / frontend) → use the **impeccable** skill.
- If the user wants to **revamp or change UI designs** (new looks, redesign whole screens, visual overhauls) → use the **ui-ux-pro-max** skill. Do NOT mix the two up.
- If any required skill is NOT available, tell the user how to install it — and offer/attempt to install it for them.

## 11. UX engagement when building frontend

- Whenever working on the frontend, bring UX sense: imagine the actual end-user and what they'd want to see in the product (e.g. if it's a bank dashboard, think like a banker/customer of that bank).
- It doesn't have to be a bank use case — for ANY project, be fully engaged with the idea and design what a real user of that product would find valuable, intuitive, and delightful.
- Put yourself in the user's seat; the UI should serve the story, not just render data.

## 12. Backend security skill (backend-security-coder)

- **Use PROACTIVELY** for any backend security implementation or security code review.
- Also covers: secure auth (JWT/OAuth/MFA/sessions/passwords), API security (rate limiting, input validation, payload/type limits), DB security (parameterized queries, user privilege separation), secure error handling & logging (no sensitive info leakage), secret management, HTTP security headers + CORS/CSRF/SSRF, cookie security (HttpOnly/SameSite), external-request allowlisting, vulnerability monitoring.
- **Risk: critical · Source: community.**
- **Do NOT use** when the task is unrelated to backend security.
- **Workflow:** clarify goals/constraints → apply best practices → validate outcomes → actionable steps; for deep detail consult `resources/implementation-playbook.md`.
- **Core behaviors:** allowlist validation, defense-in-depth, parameterized queries only, never leak secrets/sensitive data in errors or logs, least privilege, secure defaults, fail secure, audit logging.
- **vs security-auditor:** use `backend-security-coder` to WRITE secure backend code; use `security-auditor` for high-level audits, threat modeling, compliance reviews — don't mix the two up.