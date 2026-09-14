# agentic-image

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Release](https://img.shields.io/badge/release-v0.0.1-blue)
![Status](https://img.shields.io/badge/status-working%20experimental-orange)

An agent-executable local image generation studio running the open-weights
**Ideogram 4** model on your own GPU — orchestrated end to end by coding
agents such as DeepSeek Harness or Claude Code that follow [AGENTS.md](AGENTS.md)
plus five skills, instead of improvising commands.

> [!NOTE]
> Working experimental software (v0.0.1). The pipeline — interview → structured
> caption → seeded takes → browser gallery — is implemented and unit-tested,
> with a `--dry-run` path that works without a GPU. **First real Ideogram 4
> generation on a target machine is the outstanding acceptance test** (see
> [plans/IMPLEMENTATION-STATUS.md](plans/IMPLEMENTATION-STATUS.md)).

## Why Ideogram 4

Ideogram 4 ([ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4))
is a 9.3B flow-matching text-to-image model trained from scratch: best-in-class
open text rendering (signage, logos, multi-line text), explicit bounding-box
layout control, color-palette conditioning, and native 2K output — driven here
through its structured JSON caption interface.

> [!IMPORTANT]
> **Ideogram 4 weights are Non-Commercial** (gated on Hugging Face under the
> Ideogram 4 Non-Commercial license). Output from this studio is for
> personal/non-commercial use only unless you obtain separate terms.

## Requirements

- **NVIDIA CUDA GPU** (nf4 quantization, the default) — 12 GB-class or larger
  recommended (heuristic; first real render is the acceptance test). MPS/CPU
  can run fp8, untested here and slow.
- **Python 3.12+** — core scripts are stdlib-only; the model runtime lives in
  its own venv (`scripts/setup_ideogram.py`)
- **git + bash** — one-time upstream fetch (`scripts/fetch_upstream.sh`)
- [`uv`](https://docs.astral.sh/uv/) — only for the test/lint gates (`ruff`, `pytest`)
- **A Hugging Face account + token** — weights are gated: accept the license
  gate, export `HF_TOKEN`
- Optional: `IDEOGRAM_API_KEY` (free hosted magic-prompt expansion),
  `HIVE_*` keys (safety screening)

## Quickstart

This repo is designed to be driven by a coding agent, not by hand. From a harness
session opened in the repository root:

```text
you:  run env-setup
      ← agent audits hardware, fetches the upstream checkout, installs the
        runtime + weights, then smoke-tests generation

you:  I want to create an image
      ← agent interviews you, writes the structured caption,
        then renders takes
```

Manual equivalent (no agent):

```bash
bash scripts/fetch_upstream.sh        # pinned upstream checkout into oss/
python scripts/hardware_audit.py      # verify GPU + disk
python scripts/setup_ideogram.py      # venv + gated weights (needs HF_TOKEN)
# write studio/sessions/<your-id>/caption.json (see compose-brief skill),
python scripts/generate_take.py --session studio/sessions/<your-id> --seed 7
python scripts/serve_artifacts.py --port 8788   # gallery + result links
```

Images live in `studio/sessions/<image-id>/` — `brief.md`, `prompt.txt`,
`caption.json`, `takes/*.png` masters, and `review.json` when judged. Point a
browser at the artifact server (`/view/<session>/takes/take-01.png`) from any
device to see each take.

## Architecture

Two agent workspaces, one pipeline:

- **`studio/`** — the creation context. A lean `AGENTS.md` tells the agent to
  start composing immediately, render 1 take by default, hand over a result
  link per take, and never run quality review without asking first. Mistakes
  and user feedback are recorded in `studio/learnings/` (gitignored,
  per-machine), which every skill consults before acting.
- **Repository root** — development context: decision history in
  [`plans/`](plans/INDEX.md), deterministic JSON-out `scripts/`, and the five
  skills in [`.dsh/skills/`](.dsh/skills/) (a DeepSeek Harness-native discovery
  root) — including **model-guide**, the parameter and capability reference.

Pipeline: **compose-brief** (interview → `brief.md` + `prompt.txt` +
verified `caption.json`) → **generate-image** (`scripts/generate_take.py`
drives upstream `run_inference.py`) → **judge-quality** (alignment review +
artifact flags, ranked verdict).

[`configs/provider.toml`](configs/provider.toml) holds the single-model
registry plus Ideogram 4 defaults: quantization (`nf4` on CUDA, else `fp8`),
sampler preset (`V4_TURBO_12` drafts by default, `V4_QUALITY_48` for finals on request), and
canvas size (default 1024×1024).

## Project Status

v0.0.1 scaffold; first-GPU-render acceptance test outstanding. Progress tracked
in [plans/IMPLEMENTATION-STATUS.md](plans/IMPLEMENTATION-STATUS.md).

## Credits & Attribution

- **Ideogram 4** ([ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4),
  weights at [ideogram-ai/ideogram-4](https://huggingface.co/collections/ideogram-ai/ideogram-4)) —
  the image model. Referenced and fetched locally at runtime; nothing from it
  is redistributed here. Inference code is open source; **weights are
  Non-Commercial** (see [NOTICE](NOTICE)).
- Interaction protocol (interview flow, preview-and-confirm, background-job
  dispatch hygiene) adapted from the **agentic-music** studio scaffold.
- This is an independent community project, **not affiliated with or endorsed
  by** Ideogram.
- Model weights are **not included here** and must be downloaded separately by
  each user after accepting the license gate.

## License

Code in this repository is released under the [MIT License](LICENSE).
Third-party components and model weights remain under their own licenses — see
[NOTICE](NOTICE).
