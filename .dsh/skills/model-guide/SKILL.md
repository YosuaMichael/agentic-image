---
name: model-guide
description: >
  Parameter and capability reference for Ideogram 4, the image model this
  studio drives: quantizations, sampler presets, resolutions, caption schema,
  magic prompt, and safety. Read when a take needs a knob the first-class
  flags do not expose, or when deciding size/preset/quantization for a render.
---

# Skill: model-guide

## Identity

**Ideogram 4** — 9.3B flow-matching text-to-image DiT (fully single-stream,
34 layers), text encoder Qwen3-VL-8B (13 intermediate layers concatenated).
Upstream: [ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4),
`oss/ideogram4` after `fetch_upstream.sh` (see `docs/upstream.md` for the pin).
**Weights: Ideogram 4 Non-Commercial license — non-commercial use only.**
Inference code is open; weights are gated on HF (accept gate + `HF_TOKEN`).

## 1. Quantization (quality/VRAM tradeoff)

| Value | Weights | Device | Verdict on RTX 4090 (measured 2026-09-14) |
|---|---|---|---|
| `nf4` (default on CUDA) | bitsandbytes 4-bit pre-quantized (`ideogram-ai/ideogram-4-nf4`) | CUDA only | ✅ load ~95 s, 0.72 s/forward-eval @1MP, 19.5 GB peak. The only viable path here |
| `fp8` | weight-only e4m3 float8 transformer, activations bf16 (`ideogram-ai/ideogram-4-fp8`) | Any (no FP8 HW needed) | ❌ **Do not use on this machine**: load ~357 s, **28 s/forward-eval (39× slower)**, 30 GB peak. The fp8 kernels have no fast path in this environment. Revisit only with a dated plan change and fresh numbers |

Set per take: `python scripts/generate_take.py --session <dir> --quantization fp8`.
Switching quantization changes the pixels — takes record theirs in metadata.
Default stays `nf4` (see `plans/2026-09-14-perf-tuning.md` for the full table).

## 2. Sampler presets (speed/quality tradeoff)

| Preset | Steps | Character |
|---|---|---|
| `V4_QUALITY_48` (default) | 48 (45 @ gw=7 + 3 polish @ gw=3) | Finals; best quality |
| `V4_DEFAULT_20` | 20 (18 @ gw=7 + 2 polish @ gw=3) | Middle ground |
| `V4_TURBO_12` | 12 (11 @ gw=7 + 1 polish @ gw=3) | Fast drafts while iterating captions (~19 s diffusion @1MP) |

Custom schedules: add an entry to `ideogram4.sampler_configs.PRESETS`
upstream — then wire it here via a dated plan change (never hand-patch
`oss/`; it is gitignored upstream).

### Measured performance (RTX 4090, nf4, this machine, 2026-09-14)

Per-process pipeline load is ~95–100 s (paid once per OS process — see
batch mode below). Diffusion proper is 0.72 s per forward-eval at 1MP
(2 evals per step: conditional + unconditional branch):

| Preset | 1024×1024 diffusion | + load (single take) |
|---|---|---|
| `V4_TURBO_12` (drafts) | ~19 s | ~115 s |
| `V4_DEFAULT_20` | ~29 s | ~130 s |
| `V4_QUALITY_48` (finals) | ~72 s | ~172 s |

Scale-up is super-linear in pixels: QUALITY_48 at 1024×1536 (1.5MP) cost
~350 s diffusion (~3.7 s/eval) + load ≈ 456 s end to end. Rule: **iterate at
1024² TURBO, finish at target size QUALITY**. Batch `--takes N` pays load
once: 3× TURBO @1MP ≈ 100 + 3×19 ≈ 157 s total. Full table + fp8 numbers:
[plans/2026-09-14-perf-tuning.md](../../../plans/2026-09-14-perf-tuning.md).
Peak VRAM measured 19.5 GB (nf4, load+diffusion) — per-diffusion peak still
open (poll `nvidia-smi` mid-render next time).

## 3. Resolutions

Any `width × height` with both sides multiples of 16, each 256–2048, aspect
≤ 6:1. Noise schedule auto-adjusts per resolution.

| Use case | Size | Aspect |
|---|---|---|
| Square (default) | 1024 × 1024 | 1:1 |
| Landscape | 1536 × 1024 | 3:2 |
| Portrait | 1024 × 1536 | 2:3 |
| Widescreen | 1920 × 1088 | ~16:9 |
| Phone wallpaper | 1024 × 1792 | ~9:16 |
| Social banner | 1600 × 400 | 4:1 |
| Max quality | 2048 × 2048 | 1:1 |

Iterate at 1024-class, finish big. Oversize requests (>2048 / non-16
multiples / >6:1) are rejected by `generate_take.py` (exit 2) — fix the
request, don't crop after the fact and call it the requested size.

## 4. Caption schema (what the model reads)

Full guide: `oss/ideogram4/docs/prompting.md`. Compressed:

- **Always JSON.** Plain-text prompts fed directly degrade quality and trip
  safety false-positives. Either compose-brief writes `caption.json`, or a
  take uses `--use-magic-prompt`.
- Top level in order: `high_level_description` (recommended),
  `style_description`, **`compositional_deconstruction` (required)** with
  `background` then `elements` (both required).
- `style_description`: exactly one of `photo` (+`medium: "photograph"`) or
  `art_style`; order `aesthetics, lighting, photo|medium…, color_palette`
  (see compose-brief for exact orders); palette ≤ 16, uppercase `#RRGGBB`.
- Elements: `obj` (`type, bbox, desc[, palette≤5]`) for subjects;
  **`text` (`type, bbox, text, desc`) for in-image text** — `text` is the
  literal string rendered verbatim. `bbox` = optional
  `[y_min, x_min, y_max, x_max]` in 0–1000.
- Serialize compact (`separators=(",", ":")`, `ensure_ascii=False`).
- Pre-flight every caption: `python scripts/verify_caption.py --caption …`
  (stdlib mirror of upstream's `CaptionVerifier`, which also runs at
  generation time).

## 5. Magic prompt (plain text → JSON caption via LLM)

- `ideogram-4-v1` (default): Ideogram's FREE hosted API, needs
  `IDEOGRAM_API_KEY`. Server-side expansion, no local model.
- `claude-opus-v1` / `claude-sonnet-v1`: OpenRouter, needs
  `MAGIC_PROMPT_API_KEY`. System prompts are open source
  (`src/ideogram4/magic_prompt_system_prompts/`).
- The shipped magic prompt ≠ production ideogram.ai magic — results differ
  from the hosted product. Prefer hand-written captions for finals.

## 6. Safety (Hive)

Prompt screening (text) + output screening (image) via Hive, keys
`HIVE_TEXT_MODERATION_KEY` / `HIVE_VISUAL_MODERATION_KEY`. Upstream strongly
discourages running without them; without keys screening is DISABLED with a
stderr warning. A blocked render returns a gray "Image blocked by safety
filter" frame — record it as `safety-block` in review, don't present it as
art. False-positive rate is high for non-JSON prompts (another reason to
always send JSON).

## 7. Extra args (model-native knobs)

```bash
python scripts/generate_take.py --session studio/sessions/<id> --seed 7 \
    --extra-arg "--device cuda"
```

`--extra-arg` reaches `run_inference.py` verbatim (repeatable, shell-split).
Session-contract flags (`--prompt/--output/--seed/--width/--height/
--quantization/--sampler-preset/--device/--magic-prompt*/--hive-*`) are
rejected — use the first-class flag. Everything passed is recorded in
`take-NN.metadata.json`, so a take never lies about its inputs.

## 8. Capabilities & limits

- ✅ Photorealism, illustration, typography/poster design (best-in-class
  open text rendering), layout/bbox control, palette conditioning, native 2K
- ❌ No negative prompt field (steer via `desc` wording), no inpainting/edit
  mode wired here, no multi-image/reference conditioning in this pipeline
- ⚠️ One render at a time per GPU; seeds reproduce results per upstream
  (`--seed` is honored; cross-machine bit-identity is NOT promised)

## 9. Where the truth lives

- Wiring: `scripts/generate_take.py`, `configs/provider.toml`
- Upstream interface: `oss/ideogram4/run_inference.py`,
  `oss/ideogram4/docs/{prompting,inference}.md`,
  `oss/ideogram4/src/ideogram4/sampler_configs.py`
- Discovery: `python run_inference.py --help` (inside `oss/ideogram4`)
