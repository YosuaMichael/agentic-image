# 2026-09-14 — Perf tuning: 456 s → ~30 s iteration class

The 456 s acceptance render (QUALITY_48 @1024×1536) looked anomalous against
community figures (~72 s QUALITY_48 @1MP on a 4090). Instrumented benchmark
(`bench_ideogram4.py`, ad-hoc in %TEMP%, not committed) gave the breakdown.

## Measured numbers (RTX 4090, this machine)

| Config | Load (per process) | Diffusion | End to end |
|---|---|---|---|
| nf4 TURBO_12 @1024² | ~95 s | ~19 s (24 evals × 0.72 s) | ~115 s |
| nf4 DEFAULT_20 @1024² | — | ~29 s (est.) | ~130 s (est.) |
| nf4 QUALITY_48 @1024² | ~100 s | ~72 s (96 evals × 0.72 s) | ~172 s |
| nf4 QUALITY_48 @1024×1536 | ~100 s | ~350 s (96 evals × ~3.7 s) | ~456 s |
| nf4 batch 2× TURBO_12 @1024² | 99.5 s once | 19.2 s + 18.1 s | ~140 s total |
| fp8 TURBO_12 @1024² | **356.8 s** | **675 s** (24 evals × 28.1 s) | unusable |

Peak VRAM (torch): 19.5 GB nf4 (load+diffusion). fp8 reported 30 GB —
over physical VRAM, consistent with its broken-slow kernels here.

## Findings

1. **Diffusion matches community pace.** 0.72 s/forward-eval @1MP (2 evals per
   step: conditional + unconditional branch) reproduces the published ~72 s
   QUALITY_48 @1MP exactly. Nothing is wrong with the model math.
2. **Load (~100 s) dominates every fresh process** — text encoder (Qwen3-VL-8B,
   749 tensors) + 9.3B transformer + VAE + HF metadata round-trips. Paid once
   per OS process, so single-take dispatch can never beat ~115 s at TURBO.
3. **Resolution scales super-linearly**: 1.5× pixels (1MP → 1.5MP) cost ~5×
   diffusion (0.72 → ~3.7 s/eval). The 456 s render was load + big canvas,
   not a broken setup.
4. **fp8 is not a fallback on this machine**: 3.7× slower load, 39× slower
   diffusion, over-VRAM footprint. Its kernels have no fast path here.
   `nf4` is the only supported quantization until fresh numbers say otherwise
   (model-guide §1 records this as a verdict, not a suggestion).

## Shipped (this plan)

- `scripts/render_batch.py` (venv-resident, `render_batch_inner/v1`): loads
  the pipeline once, renders N (take, seed) pairs sequentially.
- `scripts/generate_take.py --takes N` (`generate_batch/v1`): one load for N
  takes; seeds cycle `[generation].seeds` by take number; `--seed`,
  `--use-magic-prompt`, `--extra-arg` rejected in batch mode (exit 2).
  Validated on GPU: 2× TURBO @1MP = 99.5 s + 19.2/18.1 s.
- Skill guidance rewritten around measured numbers: TURBO drafts @1MP,
  QUALITY finals, iterate small / finish big, batch multi-take requests,
  nf4-only (generate-image + model-guide).

## Answer to "~30 s"

- Single take floor @1MP TURBO: ~115 s cold (load-bound), 19 s warm.
- Practical loop: batch TURBO drafts (2–3 takes ≈ 140–160 s total), then one
  QUALITY final at target size. Per-image marginal cost after load: 19–72 s
  @1MP depending on preset.
- Open (not this plan): persistent warm worker to kill the 100 s load per
  request; per-diffusion VRAM peak (poll `nvidia-smi` mid-render); fp8
  re-evaluation after upstream kernel changes.
