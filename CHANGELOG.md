# Changelog

All notable changes to this project are documented here.
JSON-contract changes (`<name>/vN` bumps) must be noted.

## [Unreleased]

- `setup_ideogram.py` persists HF login (`login_persisted` in
  `setup_ideogram/v1`): after one setup with `HF_TOKEN`, later sessions reuse
  the cached weights with no token in their environment.
- New: `scripts/render_batch.py` (`render_batch_inner/v1`, venv-resident) and
  `generate_take.py --takes N` (`generate_batch/v1`) — one pipeline load for
  N takes. Seeds cycle by take number; `--seed`/`--use-magic-prompt`/
  `--extra-arg` rejected in batch mode.
- New: `generate_meta/v1` gains additive `python` (backend interpreter).
- Verdict: `nf4` is the only supported quantization on CUDA here; `fp8`
  measured 39× slower per eval (see `plans/2026-09-14-perf-tuning.md`).

## [0.0.1] — 2026-09-14

- Initial scaffold: single-model (Ideogram 4) studio mirroring agentic-music.
- Scripts: `hardware_audit` (`hardware_audit/v1`), `verify_caption`
  (`verify/v1`), `generate_take` (`generate/v1` + `generate_meta/v1`,
  `--dry-run`), `setup_ideogram` (`setup_ideogram/v1`), `fetch_upstream.sh`
  (`fetch_upstream/v1`), `serve_artifacts` (gallery, port 8788).
- Five skills: compose-brief, generate-image, judge-quality, env-setup,
  model-guide.
- Outstanding: first real GPU render (acceptance test) + measured VRAM.
