#!/usr/bin/env python3
"""Verify an Ideogram 4 structured caption without the model runtime.

Usage:
    python scripts/verify_caption.py --caption studio/sessions/<id>/caption.json

Stdlib-only check of the JSON caption schema documented upstream
(oss/ideogram4/docs/prompting.md): required top-level keys, key ORDER
(the model was trained on consistently ordered keys), hex palette format
(uppercase #RRGGBB), bbox ranges (0-1000, [y_min, x_min, y_max, x_max]),
and JSON serialisation hygiene (compact separators, no \\uXXXX escapes).

This is a pre-flight gate, not a substitute for upstream's CaptionVerifier
(which also runs inside the pipeline at generation time). A caption that
passes here matches the documented format; a caption that fails here will
degrade quality or trip safety false-positives.

JSON contract (stdout) — verify/v1:
    {"schema": "verify/v1", "ok": true|false, "caption": "<path>",
     "issues": [{"level": "error"|"warning", "where": "<json path>",
                 "message": "..."}],
     "stats": {"elements": N, "text_elements": M, "has_palette": bool}}

Exit codes: 0 caption valid (warnings allowed); 2 invalid caption or bad args.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HEX_RE = re.compile(r"^#[0-9A-F]{6}$")

TOP_REQUIRED = ["compositional_deconstruction"]
TOP_ORDER = ["high_level_description", "style_description",
             "compositional_deconstruction"]
STYLE_PHOTO_ORDER = ["aesthetics", "lighting", "photo", "medium",
                     "color_palette"]
STYLE_ART_ORDER = ["aesthetics", "lighting", "medium", "art_style",
                   "color_palette"]
OBJ_ORDER = ["type", "bbox", "desc", "color_palette"]
TEXT_ORDER = ["type", "bbox", "text", "desc", "color_palette"]


def _issue(issues: list, level: str, where: str, message: str) -> None:
    issues.append({"level": level, "where": where, "message": message})


def _check_order(obj: dict, expected: list[str], where: str,
                 issues: list) -> None:
    """Required keys must appear in `expected` relative order; extras allowed
    only where the schema permits (nowhere, currently) — unknown keys warn."""
    keys = list(obj.keys())
    for key in keys:
        if key not in expected:
            _issue(issues, "warning", where,
                   f"unknown key {key!r}; verifier may flag it")
    present = [k for k in expected if k in keys]
    want = [k for k in expected if k in present]
    if present != want:
        _issue(issues, "error", where,
               f"key order must be {want} (train order), got {present}")


def _check_palette(palette: object, where: str, issues: list,
                   limit: int) -> None:
    if not isinstance(palette, list):
        _issue(issues, "error", where, "color_palette must be a list")
        return
    if len(palette) > limit:
        _issue(issues, "error", where,
               f"color_palette has {len(palette)} entries (max {limit})")
    for i, color in enumerate(palette):
        if not isinstance(color, str) or not HEX_RE.match(color):
            _issue(issues, "error", f"{where}[{i}]",
                   f"{color!r} must be uppercase #RRGGBB (e.g. #1B1B2F)")


def _check_bbox(bbox: object, where: str, issues: list) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(
            isinstance(v, int) for v in bbox):
        _issue(issues, "error", where,
               "bbox must be [y_min, x_min, y_max, x_max] of ints")
        return
    y0, x0, y1, x1 = bbox
    if not all(0 <= v <= 1000 for v in bbox):
        _issue(issues, "error", where,
               "bbox coordinates must be in normalized 0-1000")
    if not (y0 < y1 and x0 < x1):
        _issue(issues, "error", where,
               "bbox must satisfy y_min < y_max and x_min < x_max")


def verify_file(path: Path) -> tuple[bool, list[dict], dict]:
    issues: list[dict] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [{"level": "error", "where": "$",
                        "message": f"cannot read caption: {exc}"}], {}
    if "\\u" in raw and not any(ord(c) > 127 for c in raw):
        _issue(issues, "warning", "$",
               "file contains \\uXXXX escapes; serialize with "
               "ensure_ascii=False and separators=(',', ':')")
    try:
        caption = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, [{"level": "error", "where": "$",
                        "message": f"not valid JSON: {exc}"}], {}
    if not isinstance(caption, dict):
        return False, [{"level": "error", "where": "$",
                        "message": "caption must be a JSON object"}], {}

    for key in TOP_REQUIRED:
        if key not in caption:
            _issue(issues, "error", "$",
                   f"missing required top-level key {key!r}")
    _check_order(caption, TOP_ORDER, "$", issues)

    hld = caption.get("high_level_description")
    if hld is not None and not (isinstance(hld, str) and hld.strip()):
        _issue(issues, "warning", "$.high_level_description",
               "present but empty; strongly recommended in every prompt")

    style = caption.get("style_description")
    if style is not None:
        if not isinstance(style, dict):
            _issue(issues, "error", "$.style_description",
                   "must be an object")
        else:
            has_photo = "photo" in style
            has_art = "art_style" in style
            if has_photo == has_art:
                _issue(issues, "error", "$.style_description",
                       "must contain exactly one of 'photo' / 'art_style'")
            order = STYLE_PHOTO_ORDER if has_photo else STYLE_ART_ORDER
            _check_order(style, order, "$.style_description", issues)
            for req in ("aesthetics", "lighting", "medium"):
                if req not in style:
                    _issue(issues, "error", "$.style_description",
                           f"missing required key {req!r}")
            if "color_palette" in style:
                _check_palette(style["color_palette"],
                               "$.style_description.color_palette",
                               issues, limit=16)

    stats = {"elements": 0, "text_elements": 0, "has_palette": bool(
        isinstance(style, dict) and "color_palette" in style)}
    decomp = caption.get("compositional_deconstruction")
    if isinstance(decomp, dict):
        if "background" not in decomp:
            _issue(issues, "error", "$.compositional_deconstruction",
                   "missing required key 'background'")
        elif not (isinstance(decomp["background"], str)
                  and decomp["background"].strip()):
            _issue(issues, "error",
                   "$.compositional_deconstruction.background",
                   "must be a non-empty string")
        if "elements" not in decomp:
            _issue(issues, "error", "$.compositional_deconstruction",
                   "missing required key 'elements'")
        elif not isinstance(decomp["elements"], list):
            _issue(issues, "error",
                   "$.compositional_deconstruction.elements",
                   "must be a list")
        else:
            keys = list(decomp.keys())
            if keys != [k for k in
                        ["background", "elements"] if k in keys]:
                _issue(issues, "error",
                       "$.compositional_deconstruction",
                       "key order must be ['background', 'elements']")
            for i, el in enumerate(decomp["elements"]):
                where = (f"$.compositional_deconstruction.elements[{i}]")
                if not isinstance(el, dict):
                    _issue(issues, "error", where, "must be an object")
                    continue
                stats["elements"] += 1
                etype = el.get("type")
                if etype == "text":
                    stats["text_elements"] += 1
                    _check_order(el, TEXT_ORDER, where, issues)
                    if "text" not in el or not (
                            isinstance(el["text"], str) and el["text"]):
                        _issue(issues, "error", where,
                               "text elements need a non-empty 'text' "
                               "(the literal string to render)")
                elif etype == "obj":
                    _check_order(el, OBJ_ORDER, where, issues)
                else:
                    _issue(issues, "error", where,
                           "type must be 'obj' or 'text'")
                if "bbox" in el:
                    _check_bbox(el["bbox"], where + ".bbox", issues)
                if "desc" not in el or not (
                        isinstance(el.get("desc"), str)
                        and el["desc"].strip()):
                    _issue(issues, "error", where,
                           "missing required non-empty 'desc'")
                if "color_palette" in el:
                    _check_palette(el["color_palette"],
                                   where + ".color_palette", issues,
                                   limit=5)

    ok = not any(i["level"] == "error" for i in issues)
    return ok, issues, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption", required=True, type=Path)
    args = parser.parse_args()

    if not args.caption.is_file():
        print(json.dumps({"schema": "verify/v1", "ok": False,
                          "caption": str(args.caption),
                          "issues": [{"level": "error", "where": "$",
                                      "message": "caption file not found"}],
                          "stats": {}}, indent=2))
        return 2
    ok, issues, stats = verify_file(args.caption)
    print(json.dumps({"schema": "verify/v1", "ok": ok,
                      "caption": str(args.caption), "issues": issues,
                      "stats": stats}, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
