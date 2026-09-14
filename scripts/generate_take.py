#!/usr/bin/env python3
"""Render seeded Ideogram 4 takes for a prepared session folder.

Usage:
    python scripts/generate_take.py --session studio/sessions/<image-id> [--seed 7]
        [--take-id N] [--config configs/provider.toml]
        [--width 1024 --height 1024] [--sampler-preset V4_QUALITY_48]
        [--quantization nf4|fp8] [--no-magic-prompt] [--dry-run]
        [--extra-arg "--device cuda"]

    # Batch: N takes, ONE pipeline load (amortises ~100 s of model loading)
    python scripts/generate_take.py --session studio/sessions/<image-id> --takes 3

Single-model dispatcher AND single-take generator in one file (there is only
one image model, so no backend routing table like agentic-music's
generate_take.py). Batch mode (--takes N>1) drives scripts/render_batch.py
instead, which keeps one pipeline resident for all N renders: the load cost
is paid once, not N times.
Reads the session's caption.json (the structured JSON caption compose-brief
wrote), pre-flights it with scripts/verify_caption.py, then drives upstream's
oss/ideogram4/run_inference.py as a subprocess. With --dry-run it writes a
small stdlib-generated placeholder PNG instead — the no-GPU smoke-test path.

Session inputs (compose-brief writes them):
    caption.json   REQUIRED — structured Ideogram 4 caption (verified first)
    brief.md       advisory — human-readable interview result (snapshotted)

Outputs per take:
    takes/take-NN.png               master image (lossless PNG)
    takes/take-NN.metadata.json     sidecar (generate_meta/v1)
    takes/take-NN.caption.json      frozen copy of the caption that rendered it
    takes/take-NN.brief.md          frozen copy of brief.md, when present

JSON contract (stdout) — generate/v1:
    {"schema": "generate/v1", "ok": true, "model": "ideogram4",
     "take": "take-01", "png": "takes/take-01.png",
     "metadata": "takes/take-01.metadata.json", "bytes": 12345,
     "width": 1024, "height": 1024, "elapsed_s": 61.2, "seed": 7,
     "sampler_preset": "V4_QUALITY_48", "quantization": "nf4",
     "magic_prompt_used": false, "dry_run": false,
     "extra_args": [...], "warnings": [...], "error": null}

Sidecar schema — generate_meta/v1 (takes/take-NN.metadata.json):
    {"schema": "generate_meta/v1", "model": "ideogram4", "take": "take-01",
     "seed": 7, "width": 1024, "height": 1024,
     "sampler_preset": "V4_QUALITY_48", "quantization": "nf4",
     "magic_prompt": {"used": false, "model": null},
     "caption_sha256": "...", "bytes": 12345, "elapsed_s": 61.2,
     "started_utc": "...", "extra_args": [...], "warnings": [...]}

Batch JSON contract (stdout, --takes N>1) — generate_batch/v1:
    {"schema": "generate_batch/v1", "ok": true, "model": "ideogram4",
     "load_s": 95.4, "takes": [<generate/v1 take objects, without schema>],
     "sampler_preset": "...", "quantization": "...", "dry_run": false,
     "error": null}

Exit codes: 0 success; 2 bad inputs/config (missing session or caption.json,
caption verification errors, unparseable --extra-arg, bad dimensions);
8 backend failure (run_inference.py rc != 0, no PNG produced).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import struct
import subprocess
import sys
import time
import tomllib
import zlib
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLER_PRESETS = ("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12")
QUANTIZATIONS = ("nf4", "fp8")

# Flags run_inference.py owns: the session file contract plus anything this
# script already exposes first-class. Agents may not smuggle them through
# --extra-arg (provenance protection: a take never lies about its inputs).
RESERVED_EXTRA_FLAGS = frozenset({
    "--prompt", "--output", "--seed", "--width", "--height",
    "--quantization", "--sampler-preset", "--magic-prompt",
    "--no-magic-prompt", "--magic-prompt-model", "--magic-prompt-key",
    "--device", "--hive-text-key", "--hive-visual-key",
    "--warn-on-caption-issues",
})


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def fail(message: str, code: int = 2, **extra: object) -> int:
    emit({"schema": "generate/v1", "ok": False, "error": message, **extra})
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
    """Read IHDR width/height from a PNG with stdlib only."""
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
    """Write a tiny valid PNG (stdlib zlib+struct) for --dry-run smoke tests."""
    rnd = seed % 256
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
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
    parser.add_argument("--seed", type=int, default=None,
                        help="Defaults to cycling [generation].seeds by take number")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "provider.toml")
    parser.add_argument("--take-id", type=int, default=None)
    parser.add_argument("--takes", type=int, default=1,
                        help="Render N takes in one process (one pipeline "
                             "load). Seeds cycle [generation].seeds; explicit "
                             "--seed, --use-magic-prompt and --extra-arg are "
                             "rejected in batch mode.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--sampler-preset", default=None,
                        choices=SAMPLER_PRESETS)
    parser.add_argument("--quantization", default=None,
                        choices=QUANTIZATIONS)
    parser.add_argument("--no-magic-prompt", action="store_true",
                        help="Feed caption.json verbatim (default: also verbatim; "
                             "this flag is accepted for upstream-CLI parity)")
    parser.add_argument("--use-magic-prompt", action="store_true",
                        help="Expand brief.md via the hosted magic-prompt API instead "
                             "of using caption.json (needs IDEOGRAM_API_KEY)")
    parser.add_argument("--magic-prompt-model", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="No GPU/model: verify caption, write a placeholder PNG")
    parser.add_argument("--warn-on-caption-issues", action="store_true",
                        help="Verification warnings do not block generation")
    parser.add_argument("--extra-arg", action="append", default=[],
                        metavar="ARGS",
                        help="Extra run_inference.py-native argument(s), shell-split "
                             "and forwarded verbatim (repeatable). Reserved flags "
                             "are rejected.")
    args = parser.parse_args()

    if not args.config.is_file():
        return fail(f"config not found: {args.config}")
    try:
        cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return fail(f"config is not valid TOML: {exc}")

    ide_cfg = cfg.get("ideogram4") or {}
    gen_cfg = cfg.get("generation") or {}
    seeds = [int(s) for s in (gen_cfg.get("seeds") or [7])]

    session: Path = args.session
    if not session.is_dir():
        return fail(f"session not found: {session}")

    caption_path = session / "caption.json"
    if not args.use_magic_prompt and not caption_path.is_file():
        return fail(f"caption.json not found in {session}; "
                    f"run compose-brief first (or --use-magic-prompt)")

    takes_dir = session / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)
    if args.takes < 1:
        return fail("--takes must be >= 1")
    batch = args.takes > 1
    if batch and args.seed is not None:
        return fail("explicit --seed is rejected with --takes N>1: omit it "
                    "to cycle [generation].seeds, or render singly")
    if batch and args.use_magic_prompt:
        return fail("--use-magic-prompt is rejected with --takes N>1: "
                    "render singly")
    if batch and args.extra_arg:
        return fail("--extra-arg is rejected with --takes N>1: render singly")
    first_num = args.take_id or _next_take_id(takes_dir)
    take_nums = list(range(first_num, first_num + args.takes))
    # Collision check for the whole range BEFORE spending GPU time.
    for num in take_nums:
        if (takes_dir / f"take-{num:02d}.png").exists():
            return fail(f"takes/take-{num:02d}.png already exists; pass "
                        f"--take-id to choose a free range")
    plan = [(f"take-{num:02d}",
             args.seed if args.seed is not None
             else seeds[(num - 1) % len(seeds)])
            for num in take_nums]

    width = args.width or int(ide_cfg.get("width", 1024))
    height = args.height or int(ide_cfg.get("height", 1024))
    if (width % 16 or height % 16 or not (256 <= width <= 2048)
            or not (256 <= height <= 2048)):
        return fail(f"dimensions must be multiples of 16 in 256-2048 "
                    f"(got {width}x{height})")
    aspect = width / height
    if not (1 / 6 <= aspect <= 6):
        return fail(f"aspect ratio {width}:{height} exceeds the 6:1 limit")

    sampler_preset = args.sampler_preset or str(
        ide_cfg.get("sampler_preset", "V4_TURBO_12"))
    if sampler_preset not in SAMPLER_PRESETS:
        return fail(f"unknown sampler preset {sampler_preset!r}")
    quantization = args.quantization or str(
        ide_cfg.get("quantization", "nf4"))
    if quantization not in QUANTIZATIONS:
        return fail(f"unknown quantization {quantization!r}")
    magic_model = args.magic_prompt_model or str(
        ide_cfg.get("magic_prompt_model", "ideogram-4-v1"))

    extra_tokens: list[str] = []
    for raw in args.extra_arg:
        try:
            tokens = shlex.split(raw)
        except ValueError as exc:
            return fail(f"--extra-arg {raw!r} is not shell-parseable: {exc}")
        for token in tokens:
            if token.split("=", 1)[0] in RESERVED_EXTRA_FLAGS:
                return fail(
                    f"--extra-arg may not set {token}: this script owns it "
                    f"(session file contract). Use the first-class flag — "
                    f"see the model-guide skill.")
        extra_tokens += tokens

    warnings: list[str] = []
    caption_text: str | None = None
    caption_sha = ""
    if not args.use_magic_prompt:
        # Pre-flight: same checks as scripts/verify_caption.py (imported so
        # the two can never drift apart).
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import verify_caption  # noqa: E402

        ok, issues, stats = verify_caption.verify_file(caption_path)
        for entry in issues:
            warnings.append(f"{entry['where']}: {entry['message']}")
        if not ok and not args.warn_on_caption_issues:
            return fail("caption.json failed verification "
                        "(--warn-on-caption-issues to override)",
                        warnings=warnings)
        raw_caption = caption_path.read_text(encoding="utf-8")
        # Compact serialisation, exactly as upstream expects.
        caption_text = json.dumps(json.loads(raw_caption),
                                  separators=(",", ":"),
                                  ensure_ascii=False)
        caption_sha = hashlib.sha256(
            caption_text.encode("utf-8")).hexdigest()

    # The model runtime lives in its own venv (torch + ideogram4 are NOT
    # installed in the driving interpreter). Prefer it; fall back to
    # sys.executable only when setup has never run.
    venv_dir = Path(os.path.expanduser(
        str(ide_cfg.get("venv_dir",
                        "~/.venvs/agentic-image-ideogram4"))))
    venv_python = venv_dir / ("Scripts/python.exe"
                              if os.name == "nt" else "bin/python")
    backend_python = (str(venv_python) if venv_python.is_file()
                      else sys.executable)

    def _finalize(take: str, seed: int, out_png: Path,
                  dims: tuple[int, int] | None, nbytes: int,
                  elapsed: float, started: str) -> dict:
        """Freeze provenance snapshots + sidecar; return the take object."""
        if caption_path.is_file():
            (takes_dir / f"{take}.caption.json").write_text(
                caption_path.read_text(encoding="utf-8"), encoding="utf-8")
        brief_path = session / "brief.md"
        if brief_path.is_file():
            (takes_dir / f"{take}.brief.md").write_text(
                brief_path.read_text(encoding="utf-8"), encoding="utf-8")
        meta = {
            "schema": "generate_meta/v1",
            "model": "ideogram4",
            "take": take,
            "seed": seed,
            "width": width,
            "height": height,
            "actual_width": dims[0] if dims else None,
            "actual_height": dims[1] if dims else None,
            "sampler_preset": sampler_preset,
            "quantization": quantization,
            "python": backend_python,
            "magic_prompt": {"used": magic_used,
                             "model": magic_model if magic_used else None},
            "caption_sha256": caption_sha or None,
            "bytes": nbytes,
            "elapsed_s": elapsed,
            "started_utc": started,
            "extra_args": extra_tokens,
            "warnings": warnings,
            "dry_run": args.dry_run,
        }
        (takes_dir / f"{take}.metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        return {"take": take, "png": f"takes/{take}.png",
                "metadata": f"takes/{take}.metadata.json", "bytes": nbytes,
                "width": width, "height": height, "elapsed_s": elapsed,
                "seed": seed, "sampler_preset": sampler_preset,
                "quantization": quantization,
                "magic_prompt_used": magic_used,
                "dry_run": args.dry_run,
                **({"extra_args": extra_tokens} if extra_tokens else {}),
                **({"warnings": warnings} if warnings else {})}

    magic_used = bool(args.use_magic_prompt)

    def _run_batch() -> int:
        """Render plan[] in one backend process via scripts/render_batch.py."""
        batch_started = datetime.now(UTC).isoformat(timespec="seconds")
        if args.dry_run:
            results = []
            for (btake, bseed) in plan:
                bout = takes_dir / f"{btake}.png"
                bt0 = time.monotonic()
                _placeholder_png(bout, bseed)
                bdims = _png_dimensions(bout)
                belapsed = round(time.monotonic() - bt0, 1)
                results.append(_finalize(
                    btake, bseed, bout, bdims, bout.stat().st_size,
                    belapsed, batch_started))
            emit({"schema": "generate_batch/v1", "ok": True,
                  "model": "ideogram4", "load_s": 0.0, "takes": results,
                  "sampler_preset": sampler_preset,
                  "quantization": quantization, "dry_run": True,
                  **({"warnings": warnings} if warnings else {}),
                  "error": None})
            return 0
        renderer = REPO_ROOT / "scripts" / "render_batch.py"
        if not renderer.is_file():
            return fail(f"{renderer} missing", model="ideogram4")
        cmd = [backend_python, str(renderer),
               "--caption", caption_text or "",
               "--out-dir", str(takes_dir),
               "--takes", ",".join(t for t, _ in plan),
               "--seeds", ",".join(str(s) for _, s in plan),
               "--width", str(width), "--height", str(height),
               "--sampler-preset", sampler_preset,
               "--quantization", quantization]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False)
        try:
            inner = json.loads(proc.stdout)
        except json.JSONDecodeError:
            sys.stderr.write(
                f"[generate_take] render_batch.py rc={proc.returncode} "
                f"did not print JSON\n"
                f"--- stdout tail ---\n"
                f"{chr(10).join(proc.stdout.splitlines()[-15:])}\n"
                f"--- stderr tail ---\n"
                f"{chr(10).join(proc.stderr.splitlines()[-15:])}\n")
            return fail("render_batch.py did not emit a "
                        "render_batch_inner/v1 JSON document "
                        f"(rc={proc.returncode})",
                        code=8, model="ideogram4")
        if not isinstance(inner, dict) or not inner.get("ok"):
            return fail("render_batch.py failed: "
                        f"{inner.get('error') if isinstance(inner, dict) else inner!r}",
                        code=8, model="ideogram4")
        results = []
        for item in inner.get("takes", []):
            bout = takes_dir / f"{item['take']}.png"
            if not bout.is_file() or bout.stat().st_size == 0:
                return fail(f"render_batch.py reported {item['take']} "
                            f"but wrote no PNG",
                            code=8, model="ideogram4")
            bdims = _png_dimensions(bout)
            results.append(_finalize(
                item["take"], item["seed"], bout, bdims,
                bout.stat().st_size, float(item.get("elapsed_s", 0.0)),
                batch_started))
        emit({"schema": "generate_batch/v1", "ok": True, "model": "ideogram4",
              "load_s": inner.get("load_s"), "takes": results,
              "sampler_preset": sampler_preset,
              "quantization": quantization, "dry_run": False,
              **({"warnings": warnings} if warnings else {}),
              "error": None})
        return 0

    if batch:
        return _run_batch()

    (take, seed) = plan[0]
    out_png = takes_dir / f"{take}.png"
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.monotonic()
    if args.dry_run:
        _placeholder_png(out_png, seed)
        dims = _png_dimensions(out_png)
        elapsed = round(time.monotonic() - t0, 1)
        result = _finalize(take, seed, out_png, dims,
                           out_png.stat().st_size, elapsed, started)
        emit({"schema": "generate/v1", "ok": True, "model": "ideogram4",
              **result, "error": None})
        return 0

    else:
        runner = REPO_ROOT / "oss" / "ideogram4" / "run_inference.py"
        if not runner.is_file():
            return fail(
                f"{runner} missing: run scripts/fetch_upstream.sh first, then "
                f"scripts/setup_ideogram.py", model="ideogram4")
        # Backend interpreter resolved above (venv preferred).
        cmd = [backend_python, str(runner),
               "--output", str(out_png),
               "--width", str(width), "--height", str(height),
               "--seed", str(seed),
               "--quantization", quantization,
               "--sampler-preset", sampler_preset]
        if magic_used:
            brief = session / "brief.md"
            prompt_src = (brief.read_text(encoding="utf-8")
                          if brief.is_file() else "")
            if not prompt_src.strip():
                return fail("--use-magic-prompt needs a non-empty brief.md")
            cmd += ["--prompt", prompt_src,
                    "--magic-prompt-model", magic_model]
        else:
            cmd += ["--prompt", caption_text or "",
                    "--no-magic-prompt"]
        cmd += extra_tokens
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False)
        if proc.returncode != 0:
            sys.stderr.write(
                f"[generate_take] run_inference.py rc={proc.returncode}\n"
                f"--- stdout tail ---\n"
                f"{chr(10).join(proc.stdout.splitlines()[-15:])}\n"
                f"--- stderr tail ---\n"
                f"{chr(10).join(proc.stderr.splitlines()[-15:])}\n")
            return fail("run_inference.py failed "
                        f"(rc={proc.returncode}); see stderr tail above",
                        code=8, model="ideogram4")
        if not out_png.is_file() or out_png.stat().st_size == 0:
            return fail("run_inference.py exited 0 but wrote no PNG",
                        code=8, model="ideogram4")
        dims = _png_dimensions(out_png)

    elapsed = round(time.monotonic() - t0, 1)
    result = _finalize(take, seed, out_png, dims,
                       out_png.stat().st_size, elapsed, started)
    emit({"schema": "generate/v1", "ok": True, "model": "ideogram4",
          **result, "error": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
