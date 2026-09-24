#!/usr/bin/env python3
"""Render Qwen-Image-2.1 takes in ONE process (one pipeline load).

Usage (internal — called by scripts/generate_qwen21_take.py, never by agents):
    <venv-python> scripts/render_qwen21.py --prompt <text>
        --out-dir <session>/takes --takes take-01 --seeds 42
        --width 1024 --height 1024 --steps 40 --guidance-scale 1.0
        --quantization full|q8_0

Loads the pipeline ONCE (full safetensors or Unsloth Q8_0 GGUF transformer +
full text encoder/VAE), then renders each (take, seed) sequentially with CUDA
synchronization around each render so per-take times are honest. Writes PNGs;
metadata/snapshots are the caller's job.

REQUIRES the provisioned qwen venv (torch + diffusers): this is the one script
that is not stdlib-only, by design — everything else stays stdlib so it can
run under any interpreter.

JSON contract (stdout) — render_qwen21_inner/v1:
    {"schema": "render_qwen21_inner/v1", "ok": true, "load_s": 95.4,
     "model": "qwen21", "quantization": "full",
     "takes": [{"take": "take-01", "seed": 42, "png": "<abs path>",
                "bytes": 123, "width": 1024, "height": 1024,
                "elapsed_s": 19.2}],
     "error": null}

Exit codes: 0 all takes rendered; 2 bad args; 8 backend failure (partial takes
may exist on disk; the caller reports per-take status).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--takes", required=True,
                        help="Comma-separated take names, e.g. take-01,take-02")
    parser.add_argument("--seeds", required=True,
                        help="Comma-separated int seeds, one per take")
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--guidance-scale", required=True, type=float)
    parser.add_argument("--quantization", required=True,
                        choices=["full", "q8_0"])
    parser.add_argument("--repo-dir", default=None,
                        help="Local snapshot dir of Qwen/Qwen-Image-2.1 "
                             "(full text encoder + VAE source)")
    parser.add_argument("--gguf-path", default=None,
                        help="Local path of the Q8_0 GGUF denoiser")
    args = parser.parse_args()

    import torch  # noqa: E402  (venv-only import; see docstring)

    takes = [t.strip() for t in args.takes.split(",") if t.strip()]
    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    except ValueError:
        emit({"schema": "render_qwen21_inner/v1", "ok": False,
              "error": f"--seeds must be comma-separated ints: {args.seeds!r}"})
        return 2
    if len(takes) != len(seeds) or not takes:
        emit({"schema": "render_qwen21_inner/v1", "ok": False,
              "error": "--takes and --seeds must be non-empty and aligned"})
        return 2

    hf_cache = Path(os.environ.get("HF_HUB_CACHE", ""))
    full_id = "Qwen/Qwen-Image-2.1"
    local_repo = args.repo_dir or str(
        hf_cache / ("models--" + full_id.replace("/", "--"))
        / "snapshots") if hf_cache else None

    def _snapshot_dir() -> str | None:
        if args.repo_dir and Path(args.repo_dir).is_dir():
            return args.repo_dir
        if local_repo and Path(local_repo).is_dir():
            snaps = sorted(Path(local_repo).iterdir())
            if snaps:
                return str(snaps[0])
        return None

    try:
        t0 = time.monotonic()
        if args.quantization == "full":
            from diffusers import QwenImage21Pipeline  # noqa: E402
            src = _snapshot_dir() or full_id
            pipe = QwenImage21Pipeline.from_pretrained(
                src, torch_dtype=torch.bfloat16)
        else:
            # Q8_0: Unsloth GGUF transformer + full text encoder/VAE.
            from diffusers import QwenImage21Pipeline  # noqa: E402
            gguf = args.gguf_path
            if not gguf or not Path(gguf).is_file():
                emit({"schema": "render_qwen21_inner/v1", "ok": False,
                      "error": f"Q8 GGUF not found: {gguf!r} "
                               f"(run scripts/setup_qwen21.py --quantization q8_0)"})
                return 2
            src = _snapshot_dir() or full_id
            # Transformer class name varies across diffusers commits; resolve
            # dynamically and report the candidate list on failure.
            import diffusers  # noqa: E402
            tcls = None
            tried = []
            for name in ("QwenImage21Transformer2DModel",
                         "QwenImageTransformer2DModel"):
                tried.append(name)
                if hasattr(diffusers, name):
                    tcls = getattr(diffusers, name)
                    break
            if tcls is None:
                avail = sorted(n for n in dir(diffusers)
                               if "ransformer" in n and "wen" in n)
                emit({"schema": "render_qwen21_inner/v1", "ok": False,
                      "error": f"no Qwen transformer class in diffusers "
                               f"(tried {tried}; available: {avail})"})
                return 8
            transformer = tcls.from_single_file(
                gguf, torch_dtype=torch.bfloat16)
            pipe = QwenImage21Pipeline.from_pretrained(
                src, transformer=transformer, torch_dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Prefer sequential CPU offload when available (fits 24 GB even full);
        # fall back to direct .to(device).
        try:
            pipe.enable_model_cpu_offload()
        except Exception:  # noqa: BLE001 - offload is best-effort
            pipe.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_s = round(time.monotonic() - t0, 1)
    except Exception as exc:  # noqa: BLE001 - backend errors are data
        emit({"schema": "render_qwen21_inner/v1", "ok": False,
              "error": f"pipeline load failed: {exc}"})
        return 8

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for take, seed in zip(takes, seeds, strict=True):
        out_png = out_dir / f"{take}.png"
        if out_png.exists():
            emit({"schema": "render_qwen21_inner/v1", "ok": False,
                  "load_s": load_s, "model": "qwen21",
                  "error": f"{out_png} already exists"})
            return 2
        try:
            t1 = time.monotonic()
            gen = torch.Generator("cuda" if torch.cuda.is_available()
                                  else "cpu").manual_seed(seed)
            image = pipe(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
            ).images[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = round(time.monotonic() - t1, 1)
            image.save(out_png)
            results.append({"take": take, "seed": seed,
                            "png": str(out_png),
                            "bytes": out_png.stat().st_size,
                            "width": args.width, "height": args.height,
                            "elapsed_s": elapsed})
        except Exception as exc:  # noqa: BLE001 - per-take failure is data
            emit({"schema": "render_qwen21_inner/v1", "ok": False,
                  "load_s": load_s, "model": "qwen21",
                  "takes": results,
                  "error": f"render failed for {take} (seed {seed}): {exc}"})
            return 8

    emit({"schema": "render_qwen21_inner/v1", "ok": True, "load_s": load_s,
          "model": "qwen21", "quantization": args.quantization,
          "takes": results, "error": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
