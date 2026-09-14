# Implementation status (living document)

## v0.0.1 scaffold — DONE (2026-09-14)

- [x] `configs/provider.toml` — single-model registry + Ideogram 4 defaults
- [x] `scripts/hardware_audit.py` (`hardware_audit/v1`, always exit 0)
- [x] `scripts/verify_caption.py` (`verify/v1`, stdlib schema/key-order/hex/bbox gate)
- [x] `scripts/generate_take.py` (`generate/v1` + `generate_meta/v1`, `--dry-run`)
- [x] `scripts/setup_ideogram.py` (`setup_ideogram/v1`, venv + gated weights)
- [x] `scripts/fetch_upstream.sh` (`fetch_upstream/v1`, pins oss/ideogram4)
- [x] `scripts/serve_artifacts.py` (gallery `/`, `/view/`, `/files/`, `/index.json`, :8788)
- [x] Five skills in `.dsh/skills/` (compose-brief, generate-image, judge-quality,
      env-setup, model-guide) + `agents/openai.yaml` launchers
- [x] `AGENTS.md` + `studio/AGENTS.md`, README, NOTICE, plans, tests, `.env.example`
- [x] No-GPU tests green (`pytest tests/`)

## Acceptance tests (outstanding)

1. **First real GPU render** on a target machine (env-setup Step 5):
   `generate/v1 ok:true`, non-empty PNG at requested dims. Record peak VRAM +
   wall time + preset/size/quantization in a new dated plan doc.
2. **Gallery link check**: `/view/<session>/takes/take-01.png` loads from a
   second device (tailscale serve) with preview + download working.
3. **Push to origin**: https://github.com/YosuaMichael/agentic-image is the
   remote; `main` contains exactly the scaffold (no sessions/weights/keys).

## Roadmap

- CI workflow (ruff + pytest), packaging polish, `assets/` sample gallery.
