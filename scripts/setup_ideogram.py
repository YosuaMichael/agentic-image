#!/usr/bin/env python3
"""One-time machine bring-up for the Ideogram 4 runtime.

Usage:
    python scripts/setup_ideogram.py [--config configs/provider.toml]
        [--quantization nf4|fp8] [--skip-weights] [--force]

Idempotent steps, in order:
  1. ensure the upstream checkout exists (oss/ideogram4; else run
     scripts/fetch_upstream.sh first — this script does NOT fetch git itself)
  2. ensure a venv at [ideogram4].venv_dir with the upstream package installed
     (pip install <src_dir>, or -e with --force semantics for reinstall)
  3. unless --skip-weights: pre-download the gated weight repos into the HF
     cache via the huggingface_hub API (needs HF_TOKEN + accepted gates;
     downloads are resumable)
  4. persist HF login (when HF_TOKEN is present) so later sessions reuse the
     disk cache with no token in their environment
  5. gate: import probe + weights-present check, reported as
     ready_for_generation

JSON contract (stdout) — setup_ideogram/v1:
    {"schema": "setup_ideogram/v1", "ok": true,
     "actions": [...], "skipped": [...],
     "venv_dir": "...", "installed_version": "0.1.0"|null,
     "weights_present": {"ideogram-ai/ideogram-4-nf4": true|false, ...},
     "login_persisted": true|false,
     "ready_for_generation": true|false,
     "warnings": [...], "error": null}

Weights live in the Hugging Face cache (gated repos — the user must accept
the license gate AND export HF_TOKEN before this step). Nothing is committed.

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


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "provider.toml")
    parser.add_argument("--quantization", default=None,
                        choices=["nf4", "fp8"],
                        help="Only pre-fetch this repo's weights")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Reinstall the package even if importable")
    parser.add_argument("--python", default=None,
                        help="Interpreter for the venv (shell-split, e.g. "
                             '"py -3.13"). Default: [ideogram4].python with '
                             "Windows fallbacks (py -3.12, py -3.13, python).")
    args = parser.parse_args()

    actions: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []

    if not args.config.is_file():
        emit({"schema": "setup_ideogram/v1", "ok": False,
              "error": f"config not found: {args.config}"})
        return 2
    try:
        cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        emit({"schema": "setup_ideogram/v1", "ok": False,
              "error": f"config is not valid TOML: {exc}"})
        return 2
    ide = cfg.get("ideogram4") or {}

    src_dir = REPO_ROOT / str(ide.get("src_dir", "oss/ideogram4"))
    if not (src_dir / "run_inference.py").is_file():
        emit({"schema": "setup_ideogram/v1", "ok": False,
              "error": f"upstream checkout missing at {src_dir}; "
                       f"run scripts/fetch_upstream.sh first"})
        return 2

    venv_dir = Path(os.path.expanduser(
        str(ide.get("venv_dir", "~/.venvs/agentic-image-ideogram4"))))
    venv_python = (venv_dir / ("Scripts/python.exe"
                               if os.name == "nt" else "bin/python"))

    # Resolve the venv interpreter: explicit --python wins, else the config
    # value, else Windows fallbacks (the Unix name python3.12 rarely exists
    # there; the py launcher does). First candidate that runs --version wins.
    candidates: list[list[str]] = []
    if args.python:
        try:
            candidates.append(shlex.split(args.python, posix=os.name != "nt"))
        except ValueError as exc:
            emit({"schema": "setup_ideogram/v1", "ok": False,
                  "error": f"--python is not shell-parseable: {exc}"})
            return 2
    candidates.append(shlex.split(str(ide.get("python", "python3.12"))))
    if os.name == "nt":
        candidates += [["py", "-3.12"], ["py", "-3.13"], ["python"]]
    python_cmd: list[str] | None = None
    probed: list[str] = []
    for cand in candidates:
        if cand in probed:
            continue
        probed.append(" ".join(cand))
        rc, _, _ = _run([*cand, "--version"], timeout=60)
        if rc == 0:
            python_cmd = cand
            break
    if python_cmd is None:
        emit({"schema": "setup_ideogram/v1", "ok": False,
              "error": f"no usable python found (tried: {', '.join(probed)}); "
                       f"install Python >=3.10 or pass --python"})
        return 8

    venv_created = venv_dir.is_dir()
    if not venv_created:
        rc, _, err = _run([*python_cmd, "-m", "venv", str(venv_dir)])
        if rc != 0:
            emit({"schema": "setup_ideogram/v1", "ok": False,
                  "error": f"venv creation failed ({' '.join(python_cmd)}): "
                           f"{err[-2000:]}"})
            return 8
        actions.append(f"created venv at {venv_dir} "
                       f"({' '.join(python_cmd)})")
    else:
        skipped.append(f"venv exists at {venv_dir}")

    def venv_run(mod_args: list[str],
                 timeout: int = 1800) -> tuple[int, str, str]:
        return _run([str(venv_python), *mod_args], timeout=timeout)

    installed_version: str | None = None
    rc, out, _ = venv_run(["-c",
                           "import importlib.metadata;"
                           "print(importlib.metadata.version('ideogram-4'))"],
                          timeout=120)
    needs_install = args.force or rc != 0
    if needs_install:
        rc, _, err = venv_run(["-m", "pip", "install", str(src_dir)])
        if rc != 0:
            emit({"schema": "setup_ideogram/v1", "ok": False,
                  "actions": actions, "skipped": skipped,
                  "error": f"pip install {src_dir} failed: {err[-2000:]}"})
            return 8
        actions.append(f"pip installed {src_dir}")
        rc, out, _ = venv_run(["-c",
                               "import importlib.metadata;"
                               "print(importlib.metadata.version"
                               "('ideogram-4'))"], timeout=120)
    else:
        skipped.append("ideogram4 package already importable in venv")
    if rc == 0:
        installed_version = out.strip().splitlines()[-1] if out.strip() else "unknown"

    hf_repos = list(ide.get("hf_repos") or [])
    if args.quantization:
        only = {"nf4": "ideogram-ai/ideogram-4-nf4",
                "fp8": "ideogram-ai/ideogram-4-fp8"}[args.quantization]
        hf_repos = [r for r in hf_repos if r == only]
    weights_present: dict[str, bool] = {}
    login_persisted = False
    # One-time persistent auth: `huggingface_hub.login()` stores the token in
    # ~/.cache/huggingface/token, so LATER sessions generate with no HF_TOKEN
    # in their environment (weights are reused from the shared disk cache;
    # only fast metadata revalidation touches the network). Download-once,
    # reuse-everytime starts here.
    if os.environ.get("HF_TOKEN"):
        persist = ("import os;"
                   "from huggingface_hub import login;"
                   "login(token=os.environ['HF_TOKEN']);"
                   "print('login persisted')")
        rc_login, _, err_login = venv_run(["-c", persist], timeout=120)
        login_persisted = rc_login == 0
        if login_persisted:
            actions.append("persisted HF login (~/.cache/huggingface/token)")
        else:
            warnings.append("HF login persistence FAILED: "
                            f"{err_login[-500:]} (sessions will keep "
                            f"needing HF_TOKEN)")
    if args.skip_weights:
        skipped.append("weight pre-fetch (--skip-weights)")
        for repo in hf_repos:
            weights_present[repo] = False
        warnings.append("weights not verified (--skip-weights); generation "
                        "will download them on first use (needs HF_TOKEN + "
                        "accepted gates)")
    else:
        if not os.environ.get("HF_TOKEN"):
            warnings.append("HF_TOKEN not set: gated weight downloads will "
                            "fail with 404/GatedRepoError. Accept the gates "
                            "and export HF_TOKEN, then re-run.")
        # Pre-fetch via the huggingface_hub Python API (no CLI entry-point
        # needed). snapshot_download warms the shared HF cache that
        # run_inference.py reads at generation time; HF_TOKEN is inherited
        # from this process' environment.
        for repo in hf_repos:
            prefetch = (
                "from huggingface_hub import snapshot_download;"
                f"snapshot_download(repo_id={repo!r});"
                f"print('prefetched {repo}')"
            )
            rc, _, err = venv_run(["-c", prefetch], timeout=7200)
            weights_present[repo] = rc == 0
            (actions if rc == 0 else warnings).append(
                f"pre-fetched {repo}" if rc == 0
                else f" weight download FAILED for {repo}: {err[-500:]}")

    ready = (installed_version is not None
             and bool(weights_present)
             and all(weights_present.values()))
    if installed_version is None:
        warnings.append("ideogram4 not importable in the venv after install")
    if not ready:
        warnings.append("runtime installed but NOT ready for generation "
                        "(see weights_present / warnings)")

    emit({"schema": "setup_ideogram/v1", "ok": True,
          "actions": actions, "skipped": skipped,
          "venv_dir": str(venv_dir),
          "python": " ".join(python_cmd),
          "installed_version": installed_version,
          "weights_present": weights_present,
          "login_persisted": login_persisted,
          "ready_for_generation": ready,
          "warnings": warnings, "error": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
