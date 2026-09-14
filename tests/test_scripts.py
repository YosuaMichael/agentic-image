"""No-GPU contract smoke tests: schemas present, CLIs behave, dry-run works."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

EXPECTED_SCHEMAS = {
    "hardware_audit.py": "hardware_audit/v1",
    "verify_caption.py": "verify/v1",
    "generate_take.py": "generate/v1",
    "setup_ideogram.py": "setup_ideogram/v1",
    "serve_artifacts.py": "artifacts-index/v1",
}

ADDITIONAL_SCHEMAS = {
    "generate_take.py": ["generate_meta/v1"],
}


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)


def test_scripts_import_cleanly() -> None:
    for script in ("hardware_audit.py", "verify_caption.py",
                   "generate_take.py", "setup_ideogram.py",
                   "serve_artifacts.py"):
        src = (SCRIPTS / script).read_text(encoding="utf-8")
        compile(src, script, "exec")


def test_schema_strings_present_in_source() -> None:
    for script, schema in EXPECTED_SCHEMAS.items():
        src = (SCRIPTS / script).read_text(encoding="utf-8")
        assert schema in src, f"{schema} missing from {script}"
    for script, schemas in ADDITIONAL_SCHEMAS.items():
        src = (SCRIPTS / script).read_text(encoding="utf-8")
        for schema in schemas:
            assert schema in src, f"{schema} missing from {script}"


def test_audit_reports_vram_bar() -> None:
    src = (SCRIPTS / "hardware_audit.py").read_text(encoding="utf-8")
    assert "SINGLE_GPU_MIN_VRAM_MIB" in src
    proc = run("hardware_audit.py")
    assert proc.returncode == 0
    doc = json.loads(proc.stdout)
    assert doc["schema"] == "hardware_audit/v1"
    assert "verdict" in doc and "gpus" in doc


def test_generate_missing_session_exits_2_with_json() -> None:
    proc = run("generate_take.py", "--session",
               "studio/sessions/does-not-exist", "--seed", "7")
    assert proc.returncode == 2
    doc = json.loads(proc.stdout)
    assert doc["schema"] == "generate/v1" and doc["ok"] is False


def test_verify_rejects_bad_caption(tmp_path: Path) -> None:
    bad = tmp_path / "caption.json"
    bad.write_text(json.dumps({"high_level_description": "x"}),
                   encoding="utf-8")
    proc = run("verify_caption.py", "--caption", str(bad))
    assert proc.returncode == 2
    doc = json.loads(proc.stdout)
    assert doc["schema"] == "verify/v1" and doc["ok"] is False


def test_verify_accepts_good_caption(tmp_path: Path) -> None:
    good = tmp_path / "caption.json"
    good.write_text(json.dumps({
        "high_level_description": "A red lighthouse at dusk.",
        "style_description": {
            "aesthetics": "serene",
            "lighting": "dusk glow",
            "photo": "35mm",
            "medium": "photograph",
            "color_palette": ["#FF6B35", "#1A659E"],
        },
        "compositional_deconstruction": {
            "background": "Rocky coast under a violet sky.",
            "elements": [
                {"type": "obj",
                 "desc": "A tall red lighthouse with a glowing lamp room."},
                {"type": "text", "text": "NORTH STAR",
                 "desc": "White serif letters on the lighthouse door."},
            ],
        }}), encoding="utf-8")
    proc = run("verify_caption.py", "--caption", str(good))
    assert proc.returncode == 0, proc.stdout
    doc = json.loads(proc.stdout)
    assert doc["ok"] is True
    assert doc["stats"] == {"elements": 2, "text_elements": 1,
                            "has_palette": True}


def test_generate_dry_run_end_to_end(tmp_path: Path) -> None:
    session = tmp_path / "20260914-000000-smoke-test"
    takes = session / "takes"
    takes.mkdir(parents=True)
    (session / "brief.md").write_text("# smoke\nA red lighthouse.",
                                      encoding="utf-8")
    (session / "prompt.txt").write_text("A red lighthouse at dusk.",
                                        encoding="utf-8")
    (session / "caption.json").write_text(json.dumps({
        "high_level_description": "A red lighthouse at dusk.",
        "style_description": {
            "aesthetics": "serene",
            "lighting": "dusk glow",
            "photo": "35mm",
            "medium": "photograph",
        },
        "compositional_deconstruction": {
            "background": "Rocky coast under a violet sky.",
            "elements": [{"type": "obj",
                          "desc": "A tall red lighthouse."}],
        }}), encoding="utf-8")
    config = REPO_ROOT / "configs" / "provider.toml"
    proc = run("generate_take.py", "--session", str(session), "--seed", "7",
               "--config", str(config), "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["schema"] == "generate/v1" and doc["ok"] is True
    assert doc["dry_run"] is True and doc["take"] == "take-01"
    png = takes / "take-01.png"
    assert png.is_file() and png.stat().st_size > 0
    assert (takes / "take-01.metadata.json").is_file()
    assert (takes / "take-01.caption.json").is_file()
    meta = json.loads((takes / "take-01.metadata.json").read_text())
    assert meta["schema"] == "generate_meta/v1"


def test_serve_builds_index_for_dry_run_session(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import serve_artifacts

    root = tmp_path / "sessions"
    takes = root / "20260914-000000-smoke-test" / "takes"
    takes.mkdir(parents=True)
    (takes / "take-01.png").write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
    (takes / "take-01.metadata.json").write_text(json.dumps({
        "seed": 7, "model": "ideogram4", "width": 1024, "height": 1024,
        "sampler_preset": "V4_TURBO_12"}), encoding="utf-8")
    index = serve_artifacts.build_index(root)
    assert index["schema"] == "artifacts-index/v1"
    assert len(index["sessions"]) == 1
    take = index["sessions"][0]["takes"][0]
    assert take["page"].endswith("/takes/take-01.png")
    assert take["page"].startswith("/view/")
    page = serve_artifacts.render_view_page(
        root, "20260914-000000-smoke-test/takes/take-01.png")
    assert page is not None and "take-01" in page
