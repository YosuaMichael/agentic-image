---
name: env-setup
description: >
  One-time machine bring-up for agentic-image: hardware audit, Qwen-Image-2.1
  venv + weights (default) or legacy Ideogram 4 checkout/venv/weights, smoke
  test. Use on a fresh machine or when the pipeline reports a missing
  runtime/weights.
---

# Skill: env-setup

Bring a machine from zero to generating images locally. **Default engine is
Qwen-Image-2.1** (Steps Q1–Q4 below); the legacy Ideogram 4 path follows after.

## Step 1 — Hardware audit

```bash
python scripts/hardware_audit.py
```

Parse `hardware_audit/v1`. Gate: `verdict.single_gpu_vram_ok == true`
(proceed), `null` (proceed with a warning), `false` (STOP and report).
Note `disks[]` free space: ≥60 GB free before Qwen weights (47 GB full +
~7.5 GB Q8 + torch).

## Step Q2 — Qwen runtime + weights setup (DEFAULT, ungated)

No browser gate, no token — Qwen weights are ungated (Qwen Research License,
non-commercial). Run as a background job (multi-GB torch + ~55 GB weights):

```bash
python scripts/setup_qwen21.py
```

Parse `setup_qwen21/v1`. Gate: `"ok": true` AND `"ready_for_generation":
true`. `--quantization full|q8_0` pre-fetches one variant;
`--skip-weights` installs the runtime only.

## Step Q3 — Qwen smoke test (no GPU taste, then the real thing)

```bash
python scripts/generate_qwen21_take.py --session studio/sessions/<any-session> --dry-run
```

Gate: `generate_qwen21/v1` with `"ok": true`, `"dry_run": true`. Then the
real acceptance test:

```bash
python scripts/generate_qwen21_take.py --session studio/sessions/<any-session> --seed 42
```

Gate: `"ok": true`, non-empty PNG at requested dims. **Record peak VRAM +
wall time (load vs diffusion) in a dated plan document.**

## Legacy Ideogram 4 path (only when an ideogram4 session needs it)

### Step 2 — Pin and verify upstream references

Fetches the gitignored upstream checkout into `oss/` (~source only, no
weights, no GPU):

```bash
bash scripts/fetch_upstream.sh   # Windows: Git Bash
```

Parse `fetch_upstream/v1`. Gate: `"ok": true`, zero
`missing_required_files`. This also regenerates `docs/upstream.md`.

## Step 3 — Gated weights access (human step)

The Ideogram 4 weights are **gated** on Hugging Face under a non-commercial
license. The USER must, once, in a browser:

1. Open https://huggingface.co/ideogram-ai/ideogram-4-nf4 (and/or
   `.../ideogram-4-fp8`) and click **Agree and access repository**.
2. Create a token at https://huggingface.co/settings/tokens and export it:
   `HF_TOKEN="hf_..."` (or `hf auth login`).

The agent cannot click through gates — ask the user to confirm this is done
before Step 4. No token, no weights (`404` / `GatedRepoError`). Once: after
Step 4 runs with `HF_TOKEN` present, `setup_ideogram.py` persists the login
to `~/.cache/huggingface/token` — later sessions reuse the disk cache with
**no token in their environment** (download once, reuse every time).

## Step 4 — Runtime + weights setup

Run as a background job (multi-GB torch download + ~model weights):

```bash
python scripts/setup_ideogram.py
```

Parse `setup_ideogram/v1`. Gate: `"ok": true` AND
`"ready_for_generation": true`. If `ok` but not ready, read `warnings[]` +
`weights_present` — usually the Step 3 gate (no/unauthenticated HF_TOKEN) or
a pip failure (report `error` verbatim; do not hand-install into the venv).
`--skip-weights` installs the runtime only; `--quantization nf4|fp8`
pre-fetches one repo.

## Step 5 — Smoke test (no GPU taste, then the real thing)

First, the stdlib-only path (works everywhere, no model):

```bash
python scripts/generate_take.py --session studio/sessions/<any-session> --dry-run
```

Gate: `generate/v1` with `"ok": true`, `"dry_run": true`, a non-empty PNG.
Then the real acceptance test — first REAL generation on this machine:

```bash
python scripts/generate_take.py --session studio/sessions/<any-session> --seed 7
```

Gate: `generate/v1` with `"ok": true` and a non-empty PNG at the requested
dimensions. **Record peak VRAM + wall time in a dated plan document** — that
measurement replaces the heuristic bar from Step 1 for this machine.

## Step 6 — Magic prompt + safety keys (optional, recommended)

- `IDEOGRAM_API_KEY` (https://ideogram.ai/platform) — free hosted magic-prompt
  expansion for `--use-magic-prompt`. Without it, hand-written caption.json
  (`--no-magic-prompt` path) is the way — no key needed.
- `HIVE_TEXT_MODERATION_KEY` / `HIVE_VISUAL_MODERATION_KEY`
  (https://thehive.ai) — upstream strongly recommends prompt + output
  screening; without them screening is DISABLED (runs print a warning).

Store in `.env` (gitignored; see `.env.example`). Never commit keys.

## Failure handling

| Symptom | Action |
|---|---|
| nvidia-smi missing | CPU/MPS fallback exists upstream (fp8) but is untested here — report, try `--dry-run` + fp8 before promising renders |
| Weight 404 / GatedRepoError | Step 3 undone: user accepts gates + exports HF_TOKEN, re-run setup |
| pip install fails | Report `error` verbatim with the pip tail; fix the script, don't hand-install |
| Generation OOMs | Retry the same caption at fp8 + V4_TURBO_12 + 1024×1024; if it still OOMs, STOP and record measurements in a dated plan doc before proposing mitigations |
| Safety-filter gray image on the smoke test | Rewrite the smoke caption as strict JSON (compose-brief); plain-text-shaped prompts trip it |
