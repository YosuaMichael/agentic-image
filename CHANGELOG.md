# Changelog

All notable changes to this project are documented here.
JSON-contract changes (`<name>/vN` bumps) must be noted.

## [0.0.1] — 2026-09-14

- Initial scaffold: single-model (Ideogram 4) studio mirroring agentic-music.
- Scripts: `hardware_audit` (`hardware_audit/v1`), `verify_caption`
  (`verify/v1`), `generate_take` (`generate/v1` + `generate_meta/v1`,
  `--dry-run`), `setup_ideogram` (`setup_ideogram/v1`), `fetch_upstream.sh`
  (`fetch_upstream/v1`), `serve_artifacts` (gallery, port 8788).
- Five skills: compose-brief, generate-image, judge-quality, env-setup,
  model-guide.
- Outstanding: first real GPU render (acceptance test) + measured VRAM.
