---
name: generate-image
description: >
  Render seeded image takes for a prepared session folder with Ideogram 4 via
  scripts/generate_take.py — never by hand-calling run_inference.py. Use after
  compose-brief has produced caption.json.
---

# Skill: generate-image

## Preconditions (verify, don't assume)

1. All artifacts exist on disk — verify by listing the session folder itself,
   never by trusting earlier write results (a merged/dropped tool invocation
   can silently lose a file):
   - `brief.md`, `prompt.txt`, `caption.json` must be non-empty.
2. Caption verifies clean:
   `python scripts/verify_caption.py --caption studio/sessions/<id>/caption.json`
   — fix `error`s before spending GPU time.
3. Runtime readiness is the generator's job to report; a missing
   checkout/venv/weights comes back as a precise `generate/v1` error, not a
   hang. (`python scripts/setup_ideogram.py` installs; never hand-install.)

## Procedure

1. **Learnings check.** Skim `studio/learnings/LEARNINGS.md` (small,
   append-only, gitignored — if missing, treat as empty) before dispatching;
   its rules override habit.
2. **Dispatch 1 take by default** (next seed from `[generation].seeds`).
   One image keeps iteration fast — batch more only after seeing the first
   result. When the user explicitly requested N, honor it.
3. **Dispatch takes SEQUENTIALLY** on a single GPU (one render at a time;
   concurrent dispatch contends for VRAM).
4. Run as a **background job** — one command per take:

   ```bash
   python scripts/generate_take.py --session studio/sessions/<image-id> --seed <seed>
   ```

   Overrides (first-class flags; see also the model-guide skill):
   `--width/--height` (multiples of 16, 256–2048, aspect ≤ 6:1),
   `--sampler-preset V4_QUALITY_48|V4_DEFAULT_20|V4_TURBO_12`,
   `--quantization nf4|fp8`, `--take-id N`,
   `--use-magic-prompt` (expand brief.md via hosted API; needs
   IDEOGRAM_API_KEY — only when no hand-written caption.json should be used),
   `--warn-on-caption-issues`, `--dry-run` (no-GPU placeholder),
   repeatable `--extra-arg` for run_inference.py-native knobs
   (e.g. `--extra-arg "--device cuda"`; reserved session-contract flags are
   rejected — use the first-class form).
5. Parse each result's `generate/v1` JSON. On success it names the written
   PNG and its sidecar `metadata.json`, plus `seed`, `sampler_preset`,
   `quantization`, `elapsed_s`, `bytes`, and any `warnings` (caption
   verifier notes — report them, don't hide them). The generator also
   freezes per-take copies (`take-NN.caption.json`, `take-NN.brief.md`) —
   provenance survives later edits, so never edit or delete those copies.
6. Gate: every requested take exists as a non-empty PNG whose dimensions
   match (`actual_width/actual_height` in metadata). If they don't match the
   request, say so — don't present it as the requested size.
7. **Report quick facts, then offer next actions.** Present a compact table
   per take: seed, size, sampler preset, quantization, wall time, file size,
   **and the result link** (`http://<host>:8788/view/<session>/takes/<take>.png`
   — the artifact server must be running; standalone fallback:
   `python scripts/serve_artifacts.py --host 0.0.0.0 --port 8788`). Then ask
   the user to choose one:
   - **Generate 3 more takes**
   - **Generate 1 more take**
   - **Run auto-review** (caption-alignment + artifact flags)
   - **Done / edit brief & caption**

   Loop back to step 3 for generation choices (seeds continue cycling);
   invoke `judge-quality` only after an explicit yes to that option.

## Learnings protocol

When the user corrects a mistake, or a take/session goes wrong in a way the
skill did not anticipate: append a dated entry to
`studio/learnings/LEARNINGS.md` (Symptom / Cause / Rule) in the same turn —
do not defer it. Create the folder/file if missing. Rules there override habit
on every future session.

## Dispatch hygiene & lost-job recovery

A malformed or merged tool invocation can echo a job id while nothing
actually registers — the take then sits "pending" forever while the GPU idles.
Guard against it:

1. **One tool call per invocation.** Never merge a background dispatch with
   any other call in a single block.
2. **Verify liveness before reporting an ETA.** After every dispatch:
   reading the job must NOT answer `unknown job`; and within ~60 s, either
   `nvidia-smi` shows high GPU utilization, or fresh files appear under
   `<session>/takes/`.
3. **Recovery protocol** when liveness fails: re-list jobs to confirm the
   loss, relaunch the identical command as a NEW background job, then repeat
   step 2. Never leave the session waiting on an id that shows no evidence
   of running.

## Cost guidance

- **Preset is the speed knob**: V4_TURBO_12 (~12 steps) for fast drafts,
  V4_DEFAULT_20 for the middle, V4_QUALITY_48 for finals. When iterating a
  caption with the user, render **TURBO drafts first** and only render
  QUALITY once the composition is approved.
- **Size costs VRAM/seconds**: 1024×1024 iterates fastest; 2048-wide finals
  cost more of both. Iterate small, finish big.
- **nf4 vs fp8**: nf4 (CUDA-only) is the quality default; fp8 runs anywhere.
  Switching quantization changes the pixels — record which take used which
  (metadata always does).

## Failure handling

| Symptom | Action |
|---|---|
| run_inference.py missing | `bash scripts/fetch_upstream.sh`, then `python scripts/setup_ideogram.py` |
| ideogram4 not importable / weights missing | `python scripts/setup_ideogram.py` (needs HF_TOKEN + accepted gates); report, don't hand-install |
| GatedRepoError / 404 on weights | User must accept the HF gates + export HF_TOKEN; re-run setup |
| Magic-prompt key error | Set IDEOGRAM_API_KEY, or drop `--use-magic-prompt` and use caption.json |
| Hive moderation rejection | Report the flagged side (prompt vs image) with reason; adjust brief/caption via compose-brief |
| Safety-filter gray image | The model blocked the render (often a plain-text-shaped prompt slipped through): rewrite as strict JSON caption via compose-brief |
| Caption verification errors | Fix in caption.json via compose-brief; `--warn-on-caption-issues` only for deliberate exceptions |
| Job id `unknown` / GPU idle after dispatch | Dispatch was silently lost: relaunch as a fresh background job and verify GPU utilization before reporting any ETA |

Never delete failed takes — rename with `_failed` suffix so evidence persists.
