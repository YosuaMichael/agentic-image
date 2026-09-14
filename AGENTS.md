# AGENTS.md — agentic-image Agent Operating Manual

You are an autonomous coding agent working in **agentic-image**: a repository
whose purpose is to let agents like you set up, run, and iterate on **fully
local image generation** with the open-weights **Ideogram 4** model end to
end, following written instructions — not improvisation.

## Two Workspaces

This repository has two distinct contexts:

- **Creating images?** Work inside the [`studio/`](studio/) workspace — its
  [`studio/AGENTS.md`](studio/AGENTS.md) is a lean creator manual, and images
  live in `studio/sessions/<image-id>/`. Do not read development plans there.
- **Developing the repository?** You are in the right place; continue below.

## Read Order (do not skip)

1. [`plans/INDEX.md`](plans/INDEX.md) — decision history index; read every `active` document.
2. [`plans/2026-09-14-initial-plan.md`](plans/2026-09-14-initial-plan.md) — objective,
   decisions D1–D6, architecture, risks, roadmap.
3. [`plans/2026-08-23-open-source-standards.md`](plans/2026-08-23-open-source-standards.md) —
   licensing policy, hard publication rules, engineering standards (adapted
   from agentic-music; same hygiene applies here).

Decisions recorded in `active` plan documents are binding. To change one, write a new dated
plan document and register it in the index — never silently contradict them.

## Operating Principles

1. **State lives in files, not chat.** All per-image state is under
   `studio/sessions/<image-id>/`. Any agent must be able to resume by reading that folder alone.
2. **Agents orchestrate; scripts execute.** Infra steps (audit, env, download, serve,
   generate, verify) go through `scripts/` CLIs with documented JSON output. Never
   freehand curl/python/pip for those steps. If a script is missing a capability, add it
   to the script — then use it.
3. **JSON contracts are interfaces.** Each script's docstring defines its stdout JSON
   schema (`<name>/vN`). Changing a schema is a breaking change: bump the version, update
   dependent skills, note it in `CHANGELOG.md`.
4. **Long operations run as managed background jobs** with health checks; never block a
   session interactively on multi-minute downloads or generations when the harness offers
   background jobs.

## Hard Rules (open-source hygiene)

1. NEVER `git add` anything under `oss/`, `studio/sessions/`, `models/`, `.venv/`,
   `.tools/` — they are gitignored; verify with `git status` before committing.
2. Never copy content out of `oss/ideogram4/` into committed files beyond short
   attributed quotes. Reference paths, fetch at runtime via
   `scripts/fetch_upstream.sh`. Upstream code is open but the weights are
   gated/non-commercial — keep that distinction visible.
3. No API keys, tokens, telemetry, or machine-specific absolute paths in committed files.
   Keys live in `.env` (gitignored; see `.env.example`).
4. This project is independent; never imply Ideogram affiliation in
   user-facing text.
5. **Ideogram 4 weights are Non-Commercial.** Never describe Ideogram 4 output
   as commercially usable.

## Repository Map

```
AGENTS.md            ← you are here (developer manual)
plans/               dated, indexed plan documents (decision history)
.dsh/skills/         the five skills — DSH-native discovery root (compose-brief ·
                      generate-image · judge-quality · env-setup · model-guide)
scripts/             deterministic JSON-out CLIs (the only way infra happens)
configs/provider.toml  model registry + Ideogram 4 defaults (quantization, preset, size)
studio/              image-creation workspace: studio/AGENTS.md + studio/sessions/
docs/upstream.md     pinned upstream revision + digests (generated)
oss/                 git-excluded upstream checkout created by fetch_upstream.sh
```

## The Pipeline

```
compose-brief      interview user → brief.md + prompt.txt + caption.json
                   (structured Ideogram 4 caption, verified)
generate-image     caption.json → N seeded takes
                   → takes/take-NN.png (+ metadata.json), via scripts/generate_take.py
judge-quality      alignment review + artifact flags → review.json (ranked verdict)
env-setup          once per machine: audit → upstream → weights → smoke test
```

Run `env-setup` first on a fresh machine. Then loop compose-brief → generate-image →
judge-quality, presenting result links to the user between iterations.

## Session Protocol

Image id format: `YYYYMMDD-HHMMSS-<slug>`. Artifacts are fixed names (see initial plan §7).
Never rename artifacts mid-session; superseded takes stay in place with `_failed` suffixes.
Root `brief.md` / `prompt.txt` / `caption.json` are the canonical *working* copies; each
generation also freezes per-take provenance snapshots (`takes/take-NN.caption.json`,
`take-NN.brief.md`) — treat those as immutable history.

## Platform Notes

- **Ideogram 4** is torch-based and runs on CUDA (nf4, default), MPS, or CPU
  (fp8). No WSL needed on Windows — native CUDA works.
- Gated weights: the user accepts the HF gates once + exports `HF_TOKEN`;
  `scripts/setup_ideogram.py` pre-fetches into the HF cache (outside the repo).
- Venv default `~/.venvs/agentic-image-ideogram4` (outside the repo; torch is multi-GB).
- Magic prompt (`ideogram-4-v1`, free hosted API) needs `IDEOGRAM_API_KEY`;
  hand-written captions need no key at all.
- Artifact server: `python scripts/serve_artifacts.py --port 8788` → gallery at
  `/`, per-image pages at `/view/<session>/takes/<take>.png`.

## Current Status

**v0.0.1 scaffold (2026-09-14).** Pipeline, skills, and scripts are implemented;
`--dry-run` + unit tests pass without a GPU. Outstanding acceptance test: first
REAL Ideogram 4 generation on a target machine (env-setup Step 5) with measured
peak VRAM + wall time recorded in a dated plan doc. Roadmap tracked in
[plans/IMPLEMENTATION-STATUS.md](plans/IMPLEMENTATION-STATUS.md).
