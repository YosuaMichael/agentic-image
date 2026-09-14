#!/usr/bin/env python3
"""Read-only artifact server for generated image sessions (web gallery).

Usage:
    python scripts/serve_artifacts.py [--root studio/sessions] [--host 127.0.0.1]
        [--port 8788] [--token SECRET]

Runs in the foreground (wrap in a persistent background job). Designed to sit
behind `tailscale serve` so other devices on the tailnet can view and download
generated images from a browser:

    tailscale serve --bg --https=8443 http://127.0.0.1:8788

Endpoints:
    /healthz                 liveness probe (always open)
    /                        HTML gallery: sessions -> takes with thumbnails
    /index.json              machine-readable session/take inventory
    /view/<session>/<...>    per-image page (preview + download + caption)
    /files/<session>/<...>   streamed artifacts (Range supported, image MIME)

This is THE result link: after generate-image finishes, the agent hands the
user a /view/ URL for the take. The server binds loopback only; remote access
is delegated to the tailnet — do NOT expose this to the public internet as-is.

JSON contract: none at runtime; logs one startup line to stdout.
Exit codes: 2 bad args; 0 clean shutdown.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent

MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".json": "application/json",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
CHUNK = 1 << 20  # 1 MiB stream chunks


def build_index(root: Path) -> dict:
    """Inventory studio/sessions/<id>/takes/*.{png,jpg,webp} + metadata."""
    sessions = []
    if root.is_dir():
        for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            takes_dir = session_dir / "takes"
            takes = []
            if takes_dir.is_dir():
                for f in sorted(takes_dir.iterdir()):
                    if f.suffix.lower() in IMAGE_SUFFIXES:
                        entry = {
                            "file": f.name,
                            "bytes": f.stat().st_size,
                            "url": f"/files/{session_dir.name}/takes/{f.name}",
                            "page": f"/view/{session_dir.name}/takes/{f.name}",
                        }
                        meta = f.with_name(f.stem + ".metadata.json")
                        if meta.is_file():
                            try:
                                m = json.loads(meta.read_text(encoding="utf-8"))
                                entry["seed"] = m.get("seed")
                                entry["model"] = m.get("model", "ideogram4")
                                if m.get("width"):
                                    entry["size"] = (
                                        f"{m.get('width')}x{m.get('height')}")
                                if m.get("sampler_preset"):
                                    entry["preset"] = m["sampler_preset"]
                            except Exception:  # noqa: BLE001 - advisory
                                pass
                        takes.append(entry)
            sessions.append({
                "session": session_dir.name,
                "has_brief": (session_dir / "brief.md").is_file(),
                "has_caption": (session_dir / "caption.json").is_file(),
                "takes": takes,
            })
    return {"schema": "artifacts-index/v1", "sessions": sessions}


def _read_text_capped(path: Path, limit: int = 20000) -> str | None:
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        if len(text) > limit:
            text = text[:limit] + "\n\n… truncated"
        return text
    except Exception:
        return None


def render_view_page(root: Path, rel: str) -> str | None:
    """Per-image landing page: preview, download, facts, caption, siblings."""
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    if not target.is_file() or target.suffix.lower() not in IMAGE_SUFFIXES:
        return None

    file_url = "/files/" + rel.replace("\\", "/")
    takes_dir = target.parent
    session_dir = takes_dir.parent
    stem = target.stem
    meta_path = takes_dir / f"{stem}.metadata.json"
    facts: list[str] = [f"file&nbsp;<code>{html.escape(target.name)}</code>",
                        f"{target.stat().st_size // 1024} KiB"]
    seed = model = preset = size = None
    if meta_path.is_file():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            seed = m.get("seed")
            model = m.get("model", "ideogram4")
            preset = m.get("sampler_preset")
            size = (f"{m.get('width')}×{m.get('height')}"
                    if m.get("width") else None)
            if m.get("elapsed_s"):
                facts.append(f"rendered in {m['elapsed_s']} s")
            if m.get("dry_run"):
                facts.append("DRY-RUN placeholder")
        except Exception:  # noqa: BLE001 - advisory
            pass
    facts.insert(1, f"seed&nbsp;<code>{seed if seed is not None else '?'}</code>")
    if size:
        facts.insert(2, f"size&nbsp;<code>{html.escape(size)}</code>")
    if preset:
        facts.append(f"preset&nbsp;<code>{html.escape(str(preset))}</code>")
    if model:
        facts.append(f"model&nbsp;<code>{html.escape(str(model))}</code>")

    # Prefer per-take frozen snapshots, fall back to session-level files.
    brief_text = (_read_text_capped(takes_dir / f"{stem}.brief.md")
                  or _read_text_capped(session_dir / "brief.md"))
    caption_raw = (_read_text_capped(takes_dir / f"{stem}.caption.json")
                   or _read_text_capped(session_dir / "caption.json"))
    caption_text = None
    if caption_raw:
        try:
            caption_text = json.dumps(json.loads(caption_raw), indent=2)
        except json.JSONDecodeError:
            caption_text = caption_raw

    siblings = []
    for f in sorted(takes_dir.glob("*")):
        if f.suffix.lower() in IMAGE_SUFFIXES and f != target:
            rel_sib = f.relative_to(root.resolve()).as_posix()
            siblings.append(
                f"<a href='/view/{rel_sib}'>"
                f"<img loading='lazy' src='/files/{rel_sib}' "
                f"alt='{html.escape(f.name)}' title='{html.escape(f.name)}'></a>"
            )

    def _block(title: str, text: str | None) -> str:
        if not text or not text.strip():
            return ""
        return (
            f"<section style='margin-top:1.5rem'>"
            f"<h2 style='font-size:1.05rem'>{html.escape(title)}</h2>"
            f"<pre style='white-space:pre-wrap;word-break:break-word;"
            f"background:#232a31;padding:1rem;border-radius:8px;"
            f"overflow:auto;margin:0.5rem 0 0'>{html.escape(text)}</pre>"
            f"</section>"
        )

    return (
        "<!doctype html><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(rel)} — agentic-image</title>"
        "<style>body{font-family:system-ui;max-width:44rem;margin:3rem auto;"
        "padding:0 1rem;background:#101418;color:#e8eaed}"
        "a{color:#8ab4f8}img.main{width:100%;border-radius:12px}"
        ".card{background:#1b2026;border-radius:12px;padding:1.5rem}"
        ".btn{display:inline-block;background:#8ab4f8;color:#101418;"
        "font-weight:600;padding:.7rem 1.4rem;border-radius:8px;"
        "text-decoration:none;margin-top:1rem}"
        ".facts span{margin-right:1rem;color:#9aa0a6}"
        ".thumbs{display:grid;grid-template-columns:repeat(auto-fill,"
        "minmax(120px,1fr));gap:.5rem;margin-top:.5rem}"
        ".thumbs img{width:100%;border-radius:8px}</style>"
        "<p><a href='/'>&larr; all sessions</a></p>"
        "<div class='card'>"
        "<h1 style='margin-top:0'>🖼️ " + html.escape(target.stem) + "</h1>"
        f"<img class='main' src='{file_url}' alt='{html.escape(target.name)}'>"
        f"<p class='facts'>{''.join(f'<span>{x}</span>' for x in facts)}</p>"
        f"<a class='btn' href='{file_url}' download>"
        f"⬇ Download {target.suffix[1:].upper()}</a>"
        "</div>"
        + _block("Brief", brief_text)
        + _block("Structured caption (caption.json) — what rendered this take",
                 caption_text)
        + (f"<div style='margin-top:1.5rem'><h3 style='font-size:0.95rem;"
            f"color:#9aa0a6'>Other takes</h3><div class='thumbs'>"
            f"{''.join(siblings)}</div></div>" if siblings else "")
    )


def render_html(index: dict) -> str:
    e = html.escape
    rows = []
    for s in index["sessions"]:
        if not s["takes"]:
            continue
        items = []
        for t in s["takes"]:
            origin = t.get("model") or "ideogram4"
            extra = (f", seed {t.get('seed', '?')}"
                     + (f", {t['size']}" if t.get("size") else "")
                     + (f", {t['preset']}" if t.get("preset") else ""))
            items.append(
                f"<a href='{e(t['page'])}'>"
                f"<img loading='lazy' src='{e(t['url'])}' "
                f"alt='{e(t['file'])}' title='{e(t['file'])} "
                f"({t['bytes'] // 1024} KiB{extra}, {e(str(origin))})'></a>"
            )
        rows.append(f"<h2>{e(s['session'])}</h2>"
                    f"<div class='grid'>{''.join(items)}</div>")
    body = "".join(rows) or "<p>No sessions with takes yet.</p>"
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>agentic-image artifacts</title>"
        "<style>body{font-family:system-ui;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;background:#101418;color:#e8eaed}"
        "a{color:#8ab4f8}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1rem}"
        ".grid img{width:100%;border-radius:12px}</style>"
        "<h1>🖼️ agentic-image artifacts</h1>" + body
    )


class Handler(BaseHTTPRequestHandler):
    root: Path = None  # type: ignore[assignment]
    token: str | None = None

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _authorized(self) -> bool:
        if self.token is None:
            return True
        qs = parse_qs(urlparse(self.path).query)
        if qs.get("token", [None])[0] == self.token:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, rel: str) -> Path | None:
        candidate = (self.root / unquote(rel)).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        return candidate

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/healthz":
            return self._send(200, b"ok\n", "text/plain")

        if not self._authorized():
            return self._send(401, b"unauthorized\n", "text/plain",
                              {"WWW-Authenticate": "Bearer"})

        if route in ("/", "/index.html"):
            return self._send(200, render_html(build_index(self.root)).encode(),
                              "text/html; charset=utf-8")

        if route == "/index.json":
            body = json.dumps(build_index(self.root), indent=2).encode()
            return self._send(200, body, MIME[".json"])

        if route.startswith("/view/"):
            rel = unquote(route[len("/view/"):])
            page = render_view_page(self.root, rel)
            if page is None:
                return self._send(404, b"not found\n", "text/plain")
            return self._send(200, page.encode("utf-8"),
                              "text/html; charset=utf-8")

        if route.startswith("/files/"):
            target = self._resolve(route[len("/files/"):])
            if target is None or not target.is_file():
                return self._send(404, b"not found\n", "text/plain")
            ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
            size = target.stat().st_size
            rng = self.headers.get("Range")
            if rng:
                start, end = self._parse_range(rng, size)
                if start is None:
                    return self._send(416, b"invalid range\n", "text/plain")
                length = end - start + 1
                with target.open("rb") as fh:
                    fh.seek(start)
                    remaining = length
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Content-Range",
                                     f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    while remaining > 0:
                        chunk = fh.read(min(CHUNK, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with target.open("rb") as fh:
                while chunk := fh.read(CHUNK):
                    self.wfile.write(chunk)
            return

        self._send(404, b"not found\n", "text/plain")

    @staticmethod
    def _parse_range(value: str, size: int) -> tuple[int | None, int | None]:
        try:
            unit, spec = value.split("=", 1)
            if unit.strip() != "bytes":
                return None, None
            start_s, _, end_s = spec.partition("-")
            if start_s == "":
                suffix = int(end_s)
                start = max(0, size - suffix)
                end = size - 1
            else:
                start = int(start_s)
                end = min(int(end_s), size - 1) if end_s else size - 1
            if start < 0 or start > end or start >= size:
                return None, None
            return start, end
        except ValueError:
            return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=REPO_ROOT / "studio" / "sessions")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--token", default=None,
                        help="Shared secret for non-healthz routes")
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    Handler.root = root
    Handler.token = args.token

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({
        "ok": True,
        "root": str(root),
        "url": f"http://{args.host}:{args.port}",
        "tailscale_hint": (
            f"tailscale serve --bg --https=8443 http://127.0.0.1:{args.port}"
        ),
        "token_protected": args.token is not None,
    }))
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
