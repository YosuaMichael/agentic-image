# 2026-09-14 — First GPU render (acceptance test, PASSED)

First real Ideogram 4 generation on the target machine. Env-setup Step 5 gate: PASS.

## Machine

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB, driver 591.86 (Windows native CUDA, no WSL)
- Runtime: venv `~/.venvs/agentic-image-ideogram4`, interpreter `py -3.13`,
  `torch 2.11.0+cu128` (`cuda_available True`), `ideogram-4 0.1.0`
- Upstream pin: `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2` (`docs/upstream.md`)
- Weights: `ideogram-4-nf4` + `ideogram-4-fp8` pre-fetched (gated, HF_TOKEN)

## Render (take-02, `studio/sessions/20260914-000000-demo-lighthouse/`)

- Caption: hand-written `caption.json` (verified clean, 0 issues), no magic prompt
- Params: seed 7, 1024×1536 portrait, `V4_QUALITY_48`, `nf4`
- Result: `generate/v1 ok:true`, PNG 2,265,961 bytes, actual dims 1024×1536 exact
- Wall time: **456.2 s end to end (~7.6 min)** for 48 steps at 1.5 MP
- Peak VRAM: **not captured** (only idle samples taken) — next render must poll
  `nvidia-smi` during diffusion and record the peak here.

## Quality note

Strong brief alignment (red lighthouse, glowing lamp, rocky coast, dusk sky).
The door text "NORTH STAR" renders as an illegible plaque at this scale —
small-text fidelity limit, not a pipeline defect. Larger text / closer framing
for typography-critical work.

## Setup pitfalls fixed in-repo (not worked around by hand)

1. `python3.12` missing on Windows → `setup_ideogram.py` now resolves the venv
   interpreter (config → `py -3.12` → `py -3.13` → `python`) + `--python` flag.
2. `huggingface_hub.commands.hf` module path invalid → pre-fetch via
   `snapshot_download` API instead of CLI entry points.
3. `generate_take.py` drove the backend with `sys.executable` (no ideogram4
   there) → resolves the setup venv python, recorded per take as
   `metadata.python`.
4. Upstream `pip install .` pulled `torch 2.14.0+cpu` (CUDA invisible →
   `bnb 4-bit weights require a CUDA device`) → force-reinstalled
   `torch 2.11.0+cu128`. Future setups should check `cuda_available`
   immediately after install (add to `setup_ideogram.py` readiness probe).
