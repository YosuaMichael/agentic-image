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
- Qwen-Image-2.1 as default — IN PROGRESS 2026-09-24 (see
  [2026-09-24-qwen-image-2.1](2026-09-24-qwen-image-2.1.md)):
  - [x] `configs/provider.toml` — registry (`qwen21` default) + `[qwen21]`
  - [x] `scripts/setup_qwen21.py` (`setup_qwen21/v1`), `generate_qwen21_take.py`
        (`generate_qwen21/v1` + `generate_qwen21_meta/v1`), `render_qwen21.py`
        (`render_qwen21_inner/v1`); `--dry-run` passes, pytest 14/14 green
  - [x] Skills (model-guide/generate-image/env-setup/compose-brief) + studio +
        README + NOTICE + CHANGELOG updated
  - [ ] Runtime provisioned (venv + full + Q8_0) — DONE 2026-09-24
        (venv `~/.venvs/agentic-image-qwen21`, torch 2.11 cu128;
        full 47 GB + Q8_0 7.6 GB + companions VAE/Q4_XL)
  - [x] Side-by-side full vs Q8_0 (acceptance table + verdict in the plan doc;
        gallery `/view/20260924-000000-qwen21-compare/takes/take-0{1,2}.png`)
