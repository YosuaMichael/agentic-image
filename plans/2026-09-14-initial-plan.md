# 2026-09-14 — Initial plan: agentic-image studio for Ideogram 4

## Objective

Mirror the `agentic-music` studio scaffold for **image generation** with
**Ideogram 4** (`ideogram-oss/ideogram4`): an agentic framework (skills +
deterministic scripts + studio workspace + artifact gallery) that runs
interactively inside DeepSeek Harness, hands the user a viewable link per
render, and pushes to https://github.com/YosuaMichael/agentic-image.

## Decisions

- **D1 — Single model, no ask-once gate.** agentic-music asks once per song
  which of two models to use. Here there is exactly one image model
  (Ideogram 4), so compose-brief goes straight to the interview. The per-take
  axes are quantization (`nf4`/`fp8`), sampler preset, and canvas size —
  all recorded in `take-NN.metadata.json`.
- **D2 — Upstream is invoked, never vendored.** `oss/ideogram4` is a
  gitignored checkout (`scripts/fetch_upstream.sh`); generation shells out to
  its `run_inference.py`. Only short attributed excerpts (flag names, schema
  shapes) appear in our docs. Rationale: license hygiene + upstream moves fast.
- **D3 — Scripts are stdlib-only with versioned JSON contracts.**
  `generate/v1`, `verify/v1`, `hardware_audit/v1`, `setup_ideogram/v1`,
  `serve` (no contract, startup line), `review/v1` (written by the agent, not
  a script). Exit codes: 0 ok · 2 bad input/config · 8 backend failure
  (audit always exits 0 — facts, not gates).
- **D4 — Caption discipline.** The model reads structured JSON only.
  compose-brief hand-writes `caption.json` (default, offline, no key);
  `--use-magic-prompt` is the fallback (needs `IDEOGRAM_API_KEY`).
  `scripts/verify_caption.py` pre-flights key order/hex/bbox/serialisation
  as a mirror of upstream's `CaptionVerifier`.
- **D5 — Result links are the deliverable.** `scripts/serve_artifacts.py`
  (port **8788**, distinct from agentic-music's 8787) serves a gallery (`/`),
  machine inventory (`/index.json`), per-image pages (`/view/…` with preview
  + download + caption), and streamed files (`/files/…`, Range-capable).
  Every finished take is reported with its `/view/` URL.
- **D6 — Non-commercial weights stay visible.** Ideogram 4 weights are gated
  + Non-Commercial. The gate/licence is surfaced in README, NOTICE, AGENTS.md,
  studio/AGENTS.md, and setup warnings — never described as commercial-use.

## Architecture

```
compose-brief  → studio/sessions/<id>/{brief.md, prompt.txt, caption.json}
generate-image → scripts/generate_take.py → takes/take-NN.png + .metadata.json
                                              + frozen .caption.json/.brief.md
judge-quality  → review.json (review/v1, ranked, exactly one recommendation)
env-setup      → audit → fetch_upstream → gate(HF_TOKEN, human) → setup → smoke
studio gallery ← scripts/serve_artifacts.py (:8788, tailscale-friendly)
```

## Session artifacts (§7)

`brief.md` · `prompt.txt` · `caption.json` (working copies at session root) ·
`takes/take-NN.png` · `takes/take-NN.metadata.json` (`generate_meta/v1`) ·
`takes/take-NN.caption.json` + `take-NN.brief.md` (frozen provenance) ·
`review.json` (`review/v1`) when judged. Id `YYYYMMDD-HHMMSS-<slug>`.

## Risks

1. **VRAM unknown on target machines** (9.3B DiT + 8B text encoder). Mitigation:
   12 GB heuristic bar, fp8 fallback, TURBO/small-canvas iteration, and the
   acceptance test records real peak VRAM per machine.
2. **Gated weights need a human click** (HF gate + token). Mitigation: env-setup
   Step 3 is an explicit human step; setup fails with an actionable message.
3. **Magic prompt ≠ production quality** (upstream caveat). Mitigation: hand-written
   captions are the default path.
4. **Safety-filter false positives** (esp. non-JSON prompts). Mitigation: JSON-only
   discipline + `safety-block` review flag.
5. **No GPU in CI/dev machines.** Mitigation: `--dry-run` (stdlib PNG) + unit tests.

## Roadmap

- [x] v0.0.1 scaffold: configs, scripts, skills, studio, gallery, tests
- [ ] Machine acceptance: real GPU render + measured VRAM/wall time (env-setup §5)
- [ ] CI workflow (ruff + pytest on push)
- [ ] Sample gallery under `assets/` (sanitized, model-generated, NC-noted)
- [ ] Second-model support if a commercial-use open model lands (dispatcher split)
