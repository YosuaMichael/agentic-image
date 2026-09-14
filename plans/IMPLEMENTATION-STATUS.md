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

## Acceptance tests

1. **First real GPU render** — DONE 2026-09-14 (see
   [2026-09-14-first-gpu-render](2026-09-14-first-gpu-render.md)): take-02,
   1024×1536, QUALITY_48/nf4, 456.2 s end to end on RTX 4090. Peak VRAM still
   open (poll during diffusion next time).
2. **Gallery link check** — DONE 2026-09-14: `/view/<session>/takes/take-02.png`
   and gallery `/` both 200 with the take listed. Tailscale second-device
   check still open.
3. **Push to origin** — DONE (v0.0.1 + acceptance fixes to follow in §Roadmap
   commit).

## Roadmap

- CI workflow (ruff + pytest), packaging polish, `assets/` sample gallery.
