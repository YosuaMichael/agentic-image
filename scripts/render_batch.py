#!/usr/bin/env python3
"""Render N Ideogram 4 takes in ONE process (one pipeline load).

Usage (internal — called by scripts/generate_take.py, never by agents):
    <venv-python> scripts/render_batch.py --caption <compact-json>
        --out-dir <session>/takes --takes take-02,take-03 --seeds 7,42
        --width 1024 --height 1024 --sampler-preset V4_TURBO_12
        --quantization nf4

Loads the pipeline ONCE, then renders each (take, seed) sequentially with
CUDA synchronization around each render so per-take times are honest.
Writes the PNGs; metadata/snapshots are the caller's job.

REQUIRES the provisioned venv (torch + ideogram4): this is the one script
that is not stdlib-only, by design — everything else stays stdlib so it can
run under any interpreter.

JSON contract (stdout) — render_batch_inner/v1:
    {"schema": "render_batch_inner/v1", "ok": true,
     "load_s": 95.4, "model": "ideogram4",
     "takes": [{"take": "take-02", "seed": 7, "png": "<abs path>",
                "bytes": 123, "width": 1024, "height": 1536,
                "elapsed_s": 19.2}],
     "error": null}

Exit codes: 0 all takes rendered; 8 backend failure (partial takes may exist
on disk; the caller reports per-take status).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "oss" / "ideogram4" / "src"))


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption", required=True,
                        help="Compact-serialised structured caption JSON")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--takes", required=True,
                        help="Comma-separated take names, e.g. take-02,take-03")
    parser.add_argument("--seeds", required=True,
                        help="Comma-separated int seeds, one per take")
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--sampler-preset", required=True)
    parser.add_argument("--quantization", required=True,
                        choices=["nf4", "fp8"])
    args = parser.parse_args()

    import torch  # noqa: E402  (venv-only import; see docstring)
    from ideogram4 import (  # noqa: E402
        PRESETS,
        Ideogram4Pipeline,
        Ideogram4PipelineConfig,
    )

    takes = [t.strip() for t in args.takes.split(",") if t.strip()]
    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    except ValueError:
        emit({"schema": "render_batch_inner/v1", "ok": False,
              "error": f"--seeds must be comma-separated ints: {args.seeds!r}"})
        return 2
    if len(takes) != len(seeds) or not takes:
        emit({"schema": "render_batch_inner/v1", "ok": False,
              "error": "--takes and --seeds must be non-empty and aligned"})
        return 2
    if args.sampler_preset not in PRESETS:
        emit({"schema": "render_batch_inner/v1", "ok": False,
              "error": f"unknown sampler preset {args.sampler_preset!r}"})
        return 2

    repos = {"nf4": "ideogram-ai/ideogram-4-nf4",
             "fp8": "ideogram-ai/ideogram-4-fp8"}
    try:
        t0 = time.monotonic()
        pipe = Ideogram4Pipeline.from_pretrained(
            config=Ideogram4PipelineConfig(
                weights_repo=repos[args.quantization]),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        load_s = round(time.monotonic() - t0, 1)
    except Exception as exc:  # noqa: BLE001 - backend errors are data
        emit({"schema": "render_batch_inner/v1", "ok": False,
              "error": f"pipeline load failed: {exc}"})
        return 8

    preset = PRESETS[args.sampler_preset]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for take, seed in zip(takes, seeds, strict=True):
        out_png = out_dir / f"{take}.png"
        if out_png.exists():
            emit({"schema": "render_batch_inner/v1", "ok": False,
                  "load_s": load_s, "model": "ideogram4",
                  "error": f"{out_png} already exists"})
            return 2
        try:
            t1 = time.monotonic()
            images = pipe(
                args.caption,
                height=args.height,
                width=args.width,
                num_steps=preset.num_steps,
                guidance_schedule=preset.guidance_schedule,
                mu=preset.mu,
                std=preset.std,
                seed=seed,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = round(time.monotonic() - t1, 1)
            images[0].save(out_png)
            results.append({"take": take, "seed": seed,
                            "png": str(out_png),
                            "bytes": out_png.stat().st_size,
                            "width": args.width, "height": args.height,
                            "elapsed_s": elapsed})
        except Exception as exc:  # noqa: BLE001 - per-take failure is data
            emit({"schema": "render_batch_inner/v1", "ok": False,
                  "load_s": load_s, "model": "ideogram4",
                  "takes": results,
                  "error": f"render failed for {take} (seed {seed}): {exc}"})
            return 8

    emit({"schema": "render_batch_inner/v1", "ok": True, "load_s": load_s,
          "model": "ideogram4", "takes": results, "error": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
