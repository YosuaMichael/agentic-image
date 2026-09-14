#!/usr/bin/env python3
"""Audit host hardware for local Ideogram 4 image generation.

Usage:
    python scripts/hardware_audit.py

Probes GPUs (nvidia-smi), disk free space, helper tools, and WSL presence,
then emits one JSON document. Read-only: never installs or downloads.

JSON contract (stdout) — hardware_audit/v1:
    {"schema": "hardware_audit/v1", "time_utc": "...",
     "platform": {"system": "...", "release": "...", "is_wsl": bool,
                  "wsl_distro": str|null},
     "gpus": [{"index": 0, "name": "...", "memory_total_mib": 24564,
               "driver": "..."}],
     "disks": [{"mount": "C:", "free_gb": 123.4}],
     "tools": {"python": "3.12.x"|null, "uv": "..."|null, "git": ...,
               "ffmpeg": ...|null, "curl": ...|null},
     "notes": [...],
     "verdict": {"single_gpu_vram_ok": true|false|null}}

The VRAM bar is a pre-measurement heuristic, not a published requirement:
SINGLE_GPU_MIN_VRAM_MIB = 12000. Ideogram 4 is a 9.3B DiT plus a Qwen3-VL-8B
text encoder, so treat >=12 GB as "likely fits", 8-12 GB as "may fit with fp8"
(verdict null + note), and below that as false. The FIRST real generation on a
machine is the acceptance test — record its peak VRAM in a dated plan doc.

Exit codes: always 0 (audit reports facts; the caller decides the gate).
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# Heuristic bar for a single GPU (see docstring). Multi-GPU hosts are not a
# supported topology here — Ideogram 4 runs on one device.
SINGLE_GPU_MIN_VRAM_MIB = 12000
SINGLE_GPU_SOFT_VRAM_MIB = 8000


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return 127, ""


def _probe_gpus() -> list[dict]:
    rc, out = _run([
        "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpus: list[dict] = []
    if rc != 0 or not out:
        return gpus
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(float(parts[2])),
                "driver": parts[3],
            })
        except ValueError:
            continue
    return gpus


def _probe_disks() -> list[dict]:
    mounts: list[str] = []
    if platform.system() == "Windows":
        import string

        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if Path(drive).exists():
                mounts.append(f"{letter}:")
    else:
        mounts = ["/"]
        mnt_c = Path("/mnt/c")
        if mnt_c.is_dir():
            mounts.append("/mnt/c")
    disks: list[dict] = []
    for mount in mounts:
        try:
            path = mount + ("\\" if platform.system() == "Windows" else "")
            usage = shutil.disk_usage(path)
            disks.append({
                "mount": mount,
                "free_gb": round(usage.free / (1024 ** 3), 1),
            })
        except OSError:
            continue
    return disks


def _probe_tools() -> dict[str, str | None]:
    tools: dict[str, str | None] = {}
    for name, version_args in (
        ("python", ["--version"]),
        ("uv", ["--version"]),
        ("git", ["--version"]),
        ("ffmpeg", ["-version"]),
        ("curl", ["--version"]),
    ):
        if name == "python":
            tools[name] = platform.python_version()
            continue
        rc, out = _run([name, *version_args])
        tools[name] = out.splitlines()[0] if rc == 0 and out else None
    return tools


def _probe_wsl() -> tuple[bool, str | None]:
    is_wsl = "microsoft" in platform.release().lower()
    distro: str | None = None
    rc, out = _run(["wsl.exe", "--list", "--quiet"])
    if rc == 0 and out:
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        distro = lines[0] if lines else None
    return is_wsl, distro


def main() -> int:
    gpus = _probe_gpus()
    notes: list[str] = []
    if not gpus:
        notes.append("No NVIDIA GPU detected via nvidia-smi; "
                     "Ideogram 4 falls back to MPS/CPU (fp8) but will be slow.")
        verdict: bool | None = None
    else:
        best = max(g["memory_total_mib"] for g in gpus)
        if best >= SINGLE_GPU_MIN_VRAM_MIB:
            verdict = True
        elif best >= SINGLE_GPU_SOFT_VRAM_MIB:
            verdict = None
            notes.append(
                f"Largest GPU has {best} MiB (< {SINGLE_GPU_MIN_VRAM_MIB} MiB "
                "heuristic bar): fp8 quantization may still fit; treat the "
                "first generation as the acceptance test.")
        else:
            verdict = False
            notes.append(
                f"Largest GPU has {best} MiB: below the {SINGLE_GPU_SOFT_VRAM_MIB} "
                "MiB soft floor for Ideogram 4 (9.3B DiT + Qwen3-VL-8B text "
                "encoder). Generation will likely OOM on GPU; CPU offload is "
                "untested here.")

    is_wsl, distro = _probe_wsl()

    emit = {
        "schema": "hardware_audit/v1",
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "is_wsl": is_wsl,
            "wsl_distro": distro,
        },
        "gpus": gpus,
        "disks": _probe_disks(),
        "tools": _probe_tools(),
        "notes": notes,
        "verdict": {"single_gpu_vram_ok": verdict},
    }
    print(json.dumps(emit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
