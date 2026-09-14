# Open-source standards (adapted from agentic-music, same hygiene)

## Licensing policy

- Own code: MIT (`LICENSE`). Third-party code/weights keep their licenses (`NOTICE`).
- Ideogram 4 weights are **Non-Commercial + gated**: never redistribute, never
  describe output as commercially usable, never commit tokens/keys.

## Hard publication rules

1. `git status` before every commit. Never `git add` ignored paths (`oss/`,
   `studio/sessions/`, `models/`, `.venv/`, `.tools/`, `*.png/jpg`, `.env`).
2. No keys, tokens, absolute machine paths, or telemetry in committed files.
3. Upstream material: reference paths, fetch at runtime. Short attributed
   excerpts only.
4. Independent project: never imply Ideogram affiliation.

## Engineering standards

- Scripts: stdlib-only, one JSON doc on stdout, logs on stderr, docstring =
  usage + contract + exit codes. Schema changes bump `<name>/vN` + CHANGELOG.
- Skills: front-matter (`name`, `description`) + numbered steps + failure table.
- Tests: no-GPU contract tests (`tests/`); GPU acceptance is a dated plan doc,
  not CI.
- Sessions: fixed artifact names, frozen per-take provenance, `_failed`
  suffixes (never delete evidence).
