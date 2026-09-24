# agentic-image

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Release](https://img.shields.io/badge/release-v0.0.1-blue)
![Status](https://img.shields.io/badge/status-working%20experimental-orange)

An agent-executable local image generation studio running the open-weights
**Qwen-Image-2.1** model on your own GPU — orchestrated end to end by coding
agents such as DeepSeek Harness or Claude Code that follow [AGENTS.md](AGENTS.md)
plus five skills, instead of improvising commands. (Legacy Ideogram 4 engine
still included; see `plans/2026-09-24-qwen-image-2.1.md`.)

> [!NOTE]
> Working experimental software (v0.0.1). The pipeline — interview → prompt →
> seeded takes → browser gallery — is implemented and unit-tested, with a
> `--dry-run` path that works without a GPU. **First real Qwen-Image-2.1
> generation (full vs Unsloth Q8 side-by-side) is the outstanding acceptance
> test** (see [plans/IMPLEMENTATION-STATUS.md](plans/IMPLEMENTATION-STATUS.md)).

## Why Qwen-Image-2.1

Qwen-Image-2.1 ([QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1))
is a 7B single-stream diffusion transformer with a Qwen3-VL-8B text encoder:
strong text rendering (signage, logos), native transparent (RGBA) output,
unified creation + editing (up to 10 reference images), and native 2K output —
driven here through plain-text prompts via `diffusers.QwenImage21Pipeline`.

> [!IMPORTANT]
> **Qwen-Image-2.1 weights are Non-Commercial** (ungated on Hugging Face under
> the Qwen Research License — no token needed, but no commercial use). Output
> from this studio is for personal/non-commercial use only unless you obtain
> separate terms. The Unsloth Q8 GGUF quant shares the same license.

## Requirements

- **NVIDIA CUDA GPU** — 24 GB-class recommended for the full Qwen weights
  (Q8 GGUF needs less; heuristic — first real render is the acceptance test).
  CPU offload is the default path.
- **Python 3.12+** — core scripts are stdlib-only; the model runtime lives in
  its own venv (`scripts/setup_qwen21.py`)
- **git** — only needed for the legacy Ideogram upstream fetch
- [`uv`](https://docs.astral.sh/uv/) — only for the test/lint gates (`ruff`, `pytest`)
- **No Hugging Face token needed** — Qwen weights are ungated (Qwen Research
  License, non-commercial). (Legacy Ideogram weights are gated: accept the
  license gate, export `HF_TOKEN`.)
- Optional: prompt-rewriter checkpoints (`Qwen/Qwen-Image-2.1-PE-T2I`)

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
python scripts/hardware_audit.py      # verify GPU + disk
python scripts/setup_qwen21.py        # venv + ungated weights (no token needed)
# write studio/sessions/<your-id>/prompt.txt (see compose-brief skill),
python scripts/generate_qwen21_take.py --session studio/sessions/<your-id> --seed 42
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

Pipeline: **compose-brief** (interview → `brief.md` + `prompt.txt`) →
**generate-image** (`scripts/generate_qwen21_take.py` via diffusers) →
**judge-quality** (alignment review + artifact flags, ranked verdict).

[`configs/provider.toml`](configs/provider.toml) holds the model registry
(default `qwen21`, legacy `ideogram4`) plus Qwen defaults: variant (`full`
bf16 by default, `q8_0` Unsloth GGUF), 40 steps, guidance 1.0, and canvas
size (drafts 1024×1024, native up to 2048-class).

## Project Status

v0.0.1 scaffold; Qwen-Image-2.1 full-vs-Q8 acceptance test outstanding.
Progress tracked in [plans/IMPLEMENTATION-STATUS.md](plans/IMPLEMENTATION-STATUS.md).

## Credits & Attribution

- **Qwen-Image-2.1** ([QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1),
  weights at [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
  Q8 quant at [unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)) —
  the default image model. Referenced and fetched locally at runtime; nothing
  from it is redistributed here. **Weights are Non-Commercial** (Qwen Research
  License — see [NOTICE](NOTICE)). "Qwen is licensed under the Qwen RESEARCH
  LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory
  Technology Co., Ltd. All Rights Reserved."
- **Ideogram 4** ([ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4)) —
  legacy engine, kept working (see [NOTICE](NOTICE)).
- Interaction protocol (interview flow, preview-and-confirm, background-job
  dispatch hygiene) adapted from the **agentic-music** studio scaffold.
- This is an independent community project, **not affiliated with or endorsed
  by** Alibaba/Qwen or Ideogram.
- Model weights are **not included here** and must be downloaded separately by
  each user under their own licenses.

## License

Code in this repository is released under the [MIT License](LICENSE).
Third-party components and model weights remain under their own licenses — see
[NOTICE](NOTICE).
