# 2026-09-24 — Qwen-Image-2.1 quality research: params, edits, LoRAs

Question: best parameters/method for photoreal Qwen-Image-2.1 generation +
editing (user-observed: chained edits go grainy/unphotographic; first
generations hallucinate background detail).

Method: two parallel research tracks (official docs + community, via direct
fetch — web_search was down this session) + local pipeline inspection +
empirical A/B renders on this machine (RTX 4090).

## 1. What the pipeline actually offers (installed 0.41.0.dev0, verified)

- `num_inference_steps=40`, `true_cfg_scale=1.0` (no `guidance_scale` arg).
  Negative prompts are IGNORED unless `true_cfg_scale > 1` (true CFG doubles
  per-step work). "Meant to be sampled without guidance" — official default.
- `output_resolution=1024`: side length for deriving dims AND resizing
  condition images. We rendered 1536×1024 with the conditioner at 1024-class.
- `use_kv_cache=True` (within-run speed only, not quality), custom `sigmas`
  supported, `load_lora_weights` present (needs `peft`, now in the venv).
- Schedulers available: FlowMatch Euler (default) + Heun; LCM for few-step.

## 2. Official line (Qwen repo + card + diffusers PR #14804 + ComfyUI template)

40 steps euler, cfg 1, no negative prompt, native 2K aspect sizes, kv-cache
on, PE-T2I/PE-I2I prompt rewriters recommended. Editing = same 40 steps;
up to 10 refs in ONE pass; ComfyUI template runs 25 steps euler/simple and
notes "official uses 40–50".

## 3. Why our edits went grainy (confirmed mechanisms)

1. **Every edit pass re-noises from scratch** (no `strength` knob; fresh
   latents; ComfyUI template `denoise: 1`). Chaining = full regeneration × N;
   errors compound. Our chain went 4 deep (take-01→03→04→05).
2. **16× VAE round-trip per pass** (64-ch RGBA autoencoder) — decode →
   PNG → re-encode, detail bleeds each cycle.
3. **Resolution funnel**: our 1536×1024 edits conditioned through the
   default 1024-class grid → resample softness on top.
4. CFG>1 rescales by cond/noise norms → pushes contrast/grain (we stayed at
   1.0, correctly).

## 4. Empirical results (this machine)

- **output_resolution=1536 re-test** (take-06, same winter edit as take-02):
  visibly cleaner/sharper, crisper signage and snow texture. Cost: 87 s vs
  54 s (bigger conditioning grid). **Hypothesis CONFIRMED — always match the
  long side for edits.**
- **Fix LoRA v1.0 T2I** (`e-n-v-y/Qwen-Image-2.1-Fix`, 106 MB, scale 1.0,
  seed 7): loads cleanly (+~5 s), correct signage, sane cars/people.
  Anti-hallucination effect UNPROVEN at n=1 (different seed ⇒ different
  composition anyway). No regression observed.
- **Determinism note**: seed-42 Q8 runs byte-identical across processes.

## 5. Ranked recommendations (implemented →)

1. **Re-base edits on the original, one merged prompt** (never edit-of-edit;
   10-ref single pass beats chains). Workflow rule, added to generate-image.
2. **`--output-resolution <long-side>` for every edit** (wired; 1536 for
   1536×1024). Measured win.
3. **Stay at cfg 1.0, no negative prompt** for the official look; CFG 2–3 +
   negative is community lore for artifact scrub — exposed as
   `--negative-prompt` for experiments, default off.
4. **Fix LoRA available** via `--lora <file> [--lora-scale]` (peft installed;
   file in `models/loras/`). Try scale 1.0; WarmBloodAban-style edit LoRAs
   suggest 0.6–0.8 if over-cooked.
5. **Speed path (not quality)**: Viggle 6-step turbo recipe (exact sigmas +
   shipped scheduler) or Alibaba Fun-Acc LoRAs — for drafts only.
6. **No photorealism LoRA exists yet** for 2.1 (ecosystem is 4 days old);
   the popular realism LoRA targets 1.x and is incompatible. Don't chase one.
7. **Known-VAE moiré** (diamond grid, esp. skin at high res): post-process or
   Flux-VAE re-encode per community; out of scope here.

## 6. Shipped in-repo (this plan)

- `render_qwen21.py`: `--negative-prompt`, `--output-resolution`, `--lora`,
  `--lora-scale` (diffusers/full only; q8_0 rejects edit/lora with a pointer).
- `generate_qwen21_take.py`: same flags first-class, recorded in
  `generate_qwen21_meta/v1` (`negative_prompt`, `output_resolution`, `lora`,
  `lora_scale`); warns when negative prompt is set at cfg ≤ 1.
- `setup_qwen21.py`: + `peft`.
- Fix LoRA prefetched to `models/loras/` (gitignored).
- generate-image skill: re-base-don't-chain edit discipline.

## Sources

- https://github.com/QwenLM/Qwen-Image-2.1 · https://huggingface.co/Qwen/Qwen-Image-2.1
- https://github.com/huggingface/diffusers/pull/14804 · ComfyUI `image_qwen_image_2_1_{t2i,image_edit}.json` templates
- https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo · https://huggingface.co/e-n-v-y/Qwen-Image-2.1-Fix
- https://huggingface.co/WarmBloodAban/Qwen-Image-2.1-LoRAs · https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I
- https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/qwen_image_2.1.md
