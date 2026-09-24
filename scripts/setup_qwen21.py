#!/usr/bin/env python3
"""One-time machine bring-up for the Qwen-Image-2.1 runtime.

Usage:
    python scripts/setup_qwen21.py [--config configs/provider.toml]
        [--quantization full|q8_0|all] [--skip-weights] [--force] [--python ...]

Idempotent steps, in order:
  1. ensure a venv at [qwen21].venv_dir with torch (CUDA) + transformers>=5.17
     + diffusers (from [qwen21].diffusers_git, Day-0 QwenImage21Pipeline) +
     accelerate + pillow + huggingface_hub + sentencepiece
  2. unless --skip-weights: pre-download the UNGATED weights (no HF_TOKEN,
     no gate click — Qwen Research license, non-commercial):
       - full: [qwen21].full_repo snapshot into [qwen21].hf_cache
       - q8_0: [qwen21].gguf_file_q8 from [qwen21].gguf_repo into
         [qwen21].gguf_dir (denoiser-only GGUF; text encoder + VAE come from
         the full repo at render time)
  3. gate: import probe (QwenImage21Pipeline importable + CUDA visible) +
     weights-present check, reported as ready_for_generation

JSON contract (stdout) — setup_qwen21/v1:
    {"schema": "setup_qwen21/v1", "ok": true,
     "actions": [...], "skipped": [...],
     "venv_dir": "...", "installed": {"torch": "...", "diffusers": "...",
       "transformers": "...", "cuda_available": true|false},
     "weights_present": {"full": true|false, "q8_0": true|false},
     "ready_for_generation": true|false,
     "warnings": [...], "error": null}

Weights live gitignored under models/ (never committed). Nothing here needs a
token; generation reads the cache hermetically where possible.

Exit codes: 0 success (even when ready_for_generation is false — readiness is
data, not failure); 2 bad config/args; 8 install/download failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def _run(cmd: list[str], timeout: int = 1800,
         env: dict[str, str] | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False, env=env)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)


def _resolve(path_raw: str) -> Path:
    path = Path(os.path.expanduser(path_raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "provider.toml")
    parser.add_argument("--quantization", default="all",
                        choices=["full", "q8_0", "all"],
                        help="Only pre-fetch this weight variant")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Reinstall packages even if importable")
    parser.add_argument("--python", default=None,
                        help="Interpreter for the venv (shell-split). Default: "
                             "[qwen21].python with Windows fallbacks.")
    args = parser.parse_args()

    actions: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []

    if not args.config.is_file():
        emit({"schema": "setup_qwen21/v1", "ok": False,
              "error": f"config not found: {args.config}"})
        return 2
    try:
        cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        emit({"schema": "setup_qwen21/v1", "ok": False,
              "error": f"config is not valid TOML: {exc}"})
        return 2
    qw = cfg.get("qwen21") or {}
    full_repo = str(qw.get("full_repo", "Qwen/Qwen-Image-2.1"))
    gguf_repo = str(qw.get("gguf_repo", "unsloth/Qwen-Image-2.1-GGUF"))
    gguf_file = str(qw.get("gguf_file_q8", "qwen-image-2.1-Q8_0.gguf"))
    diffusers_git = str(qw.get("diffusers_git",
                              "git+https://github.com/huggingface/diffusers"))

    venv_dir = _resolve(str(qw.get("venv_dir",
                                  "~/.venvs/agentic-image-qwen21")))
    venv_python = (venv_dir / ("Scripts/python.exe"
                               if os.name == "nt" else "bin/python"))

    candidates: list[list[str]] = []
    if args.python:
        try:
            candidates.append(shlex.split(args.python, posix=os.name != "nt"))
        except ValueError as exc:
            emit({"schema": "setup_qwen21/v1", "ok": False,
                  "error": f"--python is not shell-parseable: {exc}"})
            return 2
    candidates.append(shlex.split(str(qw.get("python", "python3.12"))))
    if os.name == "nt":
        candidates += [["py", "-3.12"], ["py", "-3.13"], ["python"]]
    python_cmd: list[str] | None = None
    probed: list[str] = []
    for cand in candidates:
        if " ".join(cand) in probed:
            continue
        probed.append(" ".join(cand))
        rc, _, _ = _run([*cand, "--version"], timeout=60)
        if rc == 0:
            python_cmd = cand
            break
    if python_cmd is None:
        emit({"schema": "setup_qwen21/v1", "ok": False,
              "error": f"no usable python found (tried: {', '.join(probed)}); "
                       f"install Python >=3.10 or pass --python"})
        return 8

    if not venv_dir.is_dir():
        rc, _, err = _run([*python_cmd, "-m", "venv", str(venv_dir)])
        if rc != 0:
            emit({"schema": "setup_qwen21/v1", "ok": False,
                  "actions": actions, "skipped": skipped,
                  "error": f"venv creation failed ({' '.join(python_cmd)}): "
                           f"{err[-2000:]}"})
            return 8
        actions.append(f"created venv at {venv_dir} "
                       f"({' '.join(python_cmd)})")
    else:
        skipped.append(f"venv exists at {venv_dir}")

    hf_cache = _resolve(str(qw.get("hf_cache", "models/hf-hub")))
    gguf_dir = _resolve(str(qw.get("gguf_dir", "models/gguf")))
    hf_cache.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    def venv_run(mod_args: list[str],
                 timeout: int = 1800) -> tuple[int, str, str]:
        env = dict(os.environ)
        env["HF_HUB_CACHE"] = str(hf_cache)
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("HF_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        return _run([str(venv_python), *mod_args], timeout=timeout, env=env)

    # --- packages ---------------------------------------------------------
    probe = ("import importlib.metadata, torch, transformers;"
             "print('torch=' + torch.__version__);"
             "print('cuda=' + str(torch.cuda.is_available()));"
             "print('transformers=' + transformers.__version__);"
             "print('diffusers=' + "
             "importlib.metadata.version('diffusers'))")
    rc, out, _ = venv_run(["-c", probe], timeout=120)
    needs_install = args.force or rc != 0
    installed: dict[str, object] = {}
    if needs_install:
        # torch FIRST from the CUDA index (plain `pip install torch` pulls a
        # CPU build on Windows — same pitfall as the Ideogram setup), then the
        # rest (diffusers from git for the Day-0 QwenImage21Pipeline class).
        rc, _, err = venv_run(
            ["-m", "pip", "install", "torch",
             "--index-url", "https://download.pytorch.org/whl/cu128"],
            timeout=7200)
        if rc != 0:
            emit({"schema": "setup_qwen21/v1", "ok": False,
                  "actions": actions, "skipped": skipped,
                  "error": f"pip install torch (cu128) failed: {err[-2000:]}"})
            return 8
        actions.append("pip installed torch (cu128)")
        pkgs = ["torchvision", "transformers>=5.17", "accelerate", "pillow",
                "sentencepiece", "huggingface_hub", "gguf", diffusers_git]
        rc, _, err = venv_run(["-m", "pip", "install", *pkgs], timeout=7200)
        if rc != 0:
            emit({"schema": "setup_qwen21/v1", "ok": False,
                  "actions": actions, "skipped": skipped,
                  "error": f"pip install failed: {err[-2000:]}"})
            return 8
        actions.append("pip installed torch + transformers + diffusers(git) "
                       "+ accelerate/pillow/sentencepiece/hf_hub/gguf")
        rc, out, _ = venv_run(["-c", probe], timeout=120)
    else:
        skipped.append("qwen21 packages already importable in venv")
    if rc == 0:
        for line in out.strip().splitlines():
            if "=" in line:
                k, v = line.strip().split("=", 1)
                installed[k] = v
        # Verify the Day-0 pipeline class exists (diffusers drift guard).
        rc2, out2, _ = venv_run(
            ["-c", "from diffusers import QwenImage21Pipeline;"
                    "print('pipeline=ok')"], timeout=120)
        installed["pipeline"] = ("ok" if rc2 == 0 and "pipeline=ok" in out2
                                 else f"MISSING: {out2[-300:]}")
        if installed["pipeline"] != "ok":
            warnings.append(
                "QwenImage21Pipeline NOT importable from installed diffusers "
                "(Day-0 class moved?) — generation will fail until diffusers "
                "is updated. See plans/2026-09-24-qwen-image-2.1.md risk 1.")
    else:
        warnings.append("package probe failed after install")

    # --- weights (UNGATED — no token needed) -------------------------------
    want = {"full", "q8_0"} if args.quantization == "all" \
        else {args.quantization}
    weights_present: dict[str, bool] = {}
    if args.skip_weights:
        skipped.append("weight pre-fetch (--skip-weights)")
        for w in ("full", "q8_0"):
            weights_present[w] = False
        warnings.append("weights not verified (--skip-weights)")
    else:
        if "full" in want:
            prefetch = (
                "from huggingface_hub import snapshot_download;"
                f"snap = snapshot_download(repo_id={full_repo!r});"
                "print('prefetched ' + snap)")
            rc, _, err = venv_run(["-c", prefetch], timeout=7200)
            weights_present["full"] = rc == 0
            (actions if rc == 0 else warnings).append(
                f"pre-fetched {full_repo}" if rc == 0
                else f" weight download FAILED for {full_repo}: {err[-500:]}")
        else:
            weights_present["full"] = (
                hf_cache / ("models--" + full_repo.replace("/", "--"))
            ).is_dir()
            skipped.append("full weights (not requested)")
        if "q8_0" in want:
            dl = ("from huggingface_hub import hf_hub_download;"
                  f"p = hf_hub_download(repo_id={gguf_repo!r}, "
                  f"filename={gguf_file!r}, local_dir={str(gguf_dir)!r});"
                  "print('downloaded ' + p)")
            rc, _, err = venv_run(["-c", dl], timeout=7200)
            weights_present["q8_0"] = (
                rc == 0 and (gguf_dir / gguf_file).is_file())
            (actions if rc == 0 and weights_present["q8_0"] else warnings).append(
                f"downloaded {gguf_repo}/{gguf_file}" if weights_present["q8_0"]
                else f" GGUF download FAILED for {gguf_file}: {err[-500:]}")
        else:
            weights_present["q8_0"] = (gguf_dir / gguf_file).is_file()
            skipped.append("q8_0 GGUF (not requested)")

    ready = (installed.get("pipeline") == "ok"
             and bool(weights_present)
             and all(weights_present[w] for w in want))
    if not ready:
        warnings.append("runtime installed but NOT ready for generation "
                        "(see weights_present / warnings)")

    emit({"schema": "setup_qwen21/v1", "ok": True,
          "actions": actions, "skipped": skipped,
          "venv_dir": str(venv_dir),
          "python": " ".join(python_cmd),
          "installed": installed,
          "weights_present": weights_present,
          "ready_for_generation": ready,
          "warnings": warnings, "error": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
