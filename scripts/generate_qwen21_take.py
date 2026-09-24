#!/usr/bin/env python3
"""Render seeded Qwen-Image-2.1 takes for a prepared session folder.

Usage:
    python scripts/generate_qwen21_take.py --session studio/sessions/<image-id>
        [--seed 42] [--quantization full|q8_0] [--steps 40]
        [--width 1024 --height 1024] [--true-cfg-scale 1.0] [--dry-run]

Qwen-Image-2.1 reads PLAIN TEXT (prompt.txt) — there is no caption.json
discipline for this engine (that belongs to the legacy ideogram4 path).
The dispatcher is stdlib-only; the backend (scripts/render_qwen21.py) runs in
the qwen venv (torch + diffusers) with weights from the agreed cache.

Session inputs:
    prompt.txt   REQUIRED — the plain-text prompt the model reads
    brief.md     advisory — human-readable interview result (snapshotted)

Outputs per take:
    takes/take-NN.png               master image (lossless PNG)
    takes/take-NN.metadata.json     sidecar (generate_qwen21_meta/v1)
    takes/take-NN.prompt.txt        frozen copy of the prompt that rendered it
    takes/take-NN.brief.md          frozen copy of brief.md, when present

JSON contract (stdout) — generate_qwen21/v1:
    {"schema": "generate_qwen21/v1", "ok": true, "model": "qwen21",
     "take": "take-01", "png": "takes/take-01.png",
     "metadata": "takes/take-01.metadata.json", "bytes": 12345,
     "width": 1024, "height": 1024, "elapsed_s": 61.2, "seed": 42,
     "steps": 40, "true_cfg_scale": 1.0, "quantization": "full",
     "dry_run": false, "warnings": [...], "error": null}

Sidecar schema — generate_qwen21_meta/v1 (takes/take-NN.metadata.json):
    {"schema": "generate_qwen21_meta/v1", "model": "qwen21", ...}

Exit codes: 0 success; 2 bad inputs/config (missing session or prompt.txt,
bad dimensions); 8 backend failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import tomllib
import zlib
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

QUANTIZATIONS = ("full", "q8_0")


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def fail(message: str, code: int = 2, **extra: object) -> int:
    emit({"schema": "generate_qwen21/v1", "ok": False, "error": message,
          **extra})
    return code


def _next_take_id(takes_dir: Path) -> int:
    taken = set()
    if takes_dir.is_dir():
        for child in takes_dir.iterdir():
            if child.suffix.lower() == ".png" and child.stem.startswith("take-"):
                try:
                    taken.add(int(child.stem.split("-", 1)[1]))
                except ValueError:
                    continue
    nxt = 1
    while nxt in taken:
        nxt += 1
    return nxt


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as fh:
            header = fh.read(33)
        if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        length, ctype = struct.unpack(">I4s", header[8:16])
        if ctype != b"IHDR" or length != 13:
            return None
        width, height = struct.unpack(">II", header[16:24])
        return width, height
    except OSError:
        return None


def _placeholder_png(path: Path, seed: int, width: int = 64,
                     height: int = 64) -> None:
    rnd = seed % 256
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw += bytes([(x + rnd) % 256, (y + rnd) % 256,
                          (x + y + rnd) % 256])
    compressed = zlib.compress(bytes(raw), 6)

    def chunk(ctype: bytes, data: bytes) -> bytes:
        body = ctype + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", compressed) + chunk(b"IEND", b""))
    path.write_bytes(png)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "provider.toml")
    parser.add_argument("--take-id", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=None)
    parser.add_argument("--quantization", default=None, choices=QUANTIZATIONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.config.is_file():
        return fail(f"config not found: {args.config}")
    try:
        cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return fail(f"config is not valid TOML: {exc}")

    qw = cfg.get("qwen21") or {}
    gen_cfg = cfg.get("generation") or {}
    seeds = [int(s) for s in (gen_cfg.get("seeds") or [7])]

    session: Path = args.session
    if not session.is_dir():
        return fail(f"session not found: {session}")
    prompt_path = session / "prompt.txt"
    if not prompt_path.is_file() or not prompt_path.read_text(
            encoding="utf-8").strip():
        return fail(f"prompt.txt missing or empty in {session}; "
                    f"run compose-brief first")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    takes_dir = session / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)
    num = args.take_id or _next_take_id(takes_dir)
    take = f"take-{num:02d}"
    if (takes_dir / f"{take}.png").exists():
        return fail(f"takes/{take}.png already exists; pass --take-id")
    seed = args.seed if args.seed is not None else seeds[(num - 1) % len(seeds)]

    width = args.width or int(qw.get("width", 1024))
    height = args.height or int(qw.get("height", 1024))
    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        return fail(f"dimensions out of range (got {width}x{height})")
    steps = args.steps or int(qw.get("num_inference_steps", 40))
    if not (1 <= steps <= 200):
        return fail(f"--steps must be 1-200 (got {steps})")
    guidance = (args.true_cfg_scale
                if args.true_cfg_scale is not None
                else float(qw.get("true_cfg_scale", 1.0)))
    quantization = args.quantization or str(qw.get("quantization", "full"))
    if quantization not in QUANTIZATIONS:
        return fail(f"unknown quantization {quantization!r}")

    venv_dir = Path(os.path.expanduser(
        str(qw.get("venv_dir", "~/.venvs/agentic-image-qwen21"))))
    venv_python = venv_dir / ("Scripts/python.exe"
                              if os.name == "nt" else "bin/python")
    backend_python = (str(venv_python) if venv_python.is_file()
                      else sys.executable)

    hf_cache = Path(os.path.expanduser(str(qw.get("hf_cache", "models/hf-hub"))))
    if not hf_cache.is_absolute():
        hf_cache = REPO_ROOT / hf_cache
    full_repo = str(qw.get("full_repo", "Qwen/Qwen-Image-2.1"))
    repo_snapshot_base = (hf_cache / ("models--" + full_repo.replace("/", "--"))
                          / "snapshots")
    gguf_dir = Path(os.path.expanduser(str(qw.get("gguf_dir", "models/gguf"))))
    if not gguf_dir.is_absolute():
        gguf_dir = REPO_ROOT / gguf_dir
    gguf_path = gguf_dir / str(qw.get("gguf_file_q8",
                                     "qwen-image-2.1-Q8_0.gguf"))

    warnings: list[str] = []
    backend_env: dict[str, str] | None = None
    repo_dir_arg: str | None = None
    if not args.dry_run:
        snaps = sorted(repo_snapshot_base.iterdir()) \
            if repo_snapshot_base.is_dir() else []
        if not snaps:
            return fail(f"full weights not in the agreed cache ({hf_cache}): "
                        f"run scripts/setup_qwen21.py first (one-time download, "
                        f"no token needed)", model="qwen21")
        repo_dir_arg = str(snaps[0])
        if quantization == "q8_0" and not gguf_path.is_file():
            return fail(f"Q8 GGUF not found ({gguf_path}): run "
                        f"scripts/setup_qwen21.py --quantization q8_0",
                        model="qwen21")
        backend_env = dict(os.environ)
        backend_env["HF_HUB_CACHE"] = str(hf_cache)
        backend_env["HF_HUB_OFFLINE"] = "1"

    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.monotonic()
    out_png = takes_dir / f"{take}.png"
    load_s = 0.0
    if args.dry_run:
        _placeholder_png(out_png, seed)
    else:
        renderer = REPO_ROOT / "scripts" / "render_qwen21.py"
        cmd = [backend_python, str(renderer),
               "--prompt", prompt,
               "--out-dir", str(takes_dir),
               "--takes", take, "--seeds", str(seed),
               "--width", str(width), "--height", str(height),
               "--steps", str(steps), "--true-cfg-scale", str(guidance),
               "--quantization", quantization,
               "--repo-dir", repo_dir_arg or ""]
        if quantization == "q8_0":
            cmd += ["--gguf-path", str(gguf_path)]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, env=backend_env)
        try:
            inner = json.loads(proc.stdout)
        except json.JSONDecodeError:
            sys.stderr.write(
                f"[generate_qwen21] render_qwen21.py rc={proc.returncode} "
                f"did not print JSON\n--- stdout tail ---\n"
                f"{chr(10).join(proc.stdout.splitlines()[-15:])}\n"
                f"--- stderr tail ---\n"
                f"{chr(10).join(proc.stderr.splitlines()[-15:])}\n")
            return fail("render_qwen21.py did not emit a "
                        "render_qwen21_inner/v1 JSON document "
                        f"(rc={proc.returncode})", code=8, model="qwen21")
        if not isinstance(inner, dict) or not inner.get("ok"):
            err = inner.get("error") if isinstance(inner, dict) else inner
            return fail(f"render_qwen21.py failed: {err}", code=8,
                        model="qwen21")
        load_s = float(inner.get("load_s", 0.0))
        item = (inner.get("takes") or [{}])[0]
        elapsed = float(item.get("elapsed_s", 0.0))
        if not out_png.is_file() or out_png.stat().st_size == 0:
            return fail("render_qwen21.py reported ok but wrote no PNG",
                        code=8, model="qwen21")
    dims = _png_dimensions(out_png)
    elapsed = round(time.monotonic() - t0, 1) if args.dry_run else elapsed

    (takes_dir / f"{take}.prompt.txt").write_text(prompt, encoding="utf-8")
    brief_path = session / "brief.md"
    if brief_path.is_file():
        (takes_dir / f"{take}.brief.md").write_text(
            brief_path.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "schema": "generate_qwen21_meta/v1",
        "model": "qwen21",
        "take": take,
        "seed": seed,
        "width": width,
        "height": height,
        "actual_width": dims[0] if dims else None,
        "actual_height": dims[1] if dims else None,
        "steps": steps,
        "true_cfg_scale": guidance,
        "quantization": quantization,
        "python": backend_python,
        "prompt_sha256": prompt_sha,
        "bytes": out_png.stat().st_size,
        "elapsed_s": elapsed,
        "load_s": load_s,
        "started_utc": started,
        "warnings": warnings,
        "dry_run": args.dry_run,
    }
    (takes_dir / f"{take}.metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    emit({"schema": "generate_qwen21/v1", "ok": True, "model": "qwen21",
          "take": take, "png": f"takes/{take}.png",
          "metadata": f"takes/{take}.metadata.json",
          "bytes": out_png.stat().st_size, "width": width, "height": height,
          "elapsed_s": elapsed, "load_s": load_s, "seed": seed,
          "steps": steps, "true_cfg_scale": guidance,
          "quantization": quantization, "dry_run": args.dry_run,
          **({"warnings": warnings} if warnings else {}), "error": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
