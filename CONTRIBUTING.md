# Contributing

Short version: follow `AGENTS.md` (root manual) and
`plans/2026-08-23-open-source-standards.md`. The details:

1. Read `plans/INDEX.md` (every `active` doc) before changing behavior.
2. Infra goes through `scripts/` CLIs with versioned JSON contracts; update the
   docstring contract + `CHANGELOG.md` when a schema changes.
3. Run the gates before opening a PR: `ruff check .` and `pytest tests/`.
4. `git status` before committing — never stage `oss/`, `studio/sessions/`,
   `models/`, `.venv/`, `.tools/`, generated images, or `.env`.
5. Binding decisions change via new dated plan docs, never silently.
