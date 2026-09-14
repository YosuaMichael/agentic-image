# 2026-09-14 — Hermetic weights: repo-local cache, no login, no network

Generating in a fresh session re-asked for the HF token: weights were cached,
but every run revalidated metadata against the gated repo, and the token only
existed in the setup shell's environment. Fixed by agreeing on a repo-local
weight home and forcing the backend offline.

## Diagnosis

1. `~/.cache/huggingface` held both repos (nf4 15 GB, fp8 25.7 GB) with refs —
   downloads were never the problem; per-process *auth* was.
2. `HF_HUB_OFFLINE=1` loads failed on `AutoTokenizer(tokenizer/)`: upstream
   ships NO `tokenizer/config.json` (verified via recursive tree listing —
   only `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`),
   and this transformers version cannot fall back offline. Online, the 404
   falls back to `tokenizer_config.json` (`tokenizer_class: Qwen2Tokenizer`).
3. Fix: a one-file offline shim — `tokenizer/config.json` =
   `{"model_type": "qwen2"}` — written into the cached snapshot. Offline
   tokenizer load then succeeds (`Qwen2Tokenizer`), and full offline pipeline
   load succeeds (91–99 s, same as online).

## Shipped

- Weights relocated to `models/hf-hub/` (gitignored): `models--ideogram-ai--ideogram-4-{nf4,fp8}` moved, not re-downloaded. Verified offline load from the new home (93.4 s).
- `configs/provider.toml`: `[ideogram4].hf_cache = "models/hf-hub"` — the
  agreed location, resolved repo-relative.
- `setup_ideogram.py`: all venv calls run with `HF_HUB_CACHE` pointed at the
  agreed cache (offline flags stripped — setup stays online); prefetch writes
  the offline shim per repo (asserts a known `tokenizer_class` mapping);
  `--skip-weights` warning corrected (generation is offline and would fail,
  not download).
- `generate_take.py`: backend (`run_inference.py` + `render_batch.py`) runs
  with `HF_HUB_CACHE` + `HF_HUB_OFFLINE=1`. Cold cache fails fast (exit 2)
  pointing at setup. Covered by `test_cold_cache_fails_fast_with_setup_hint`.
- Persistent `hf auth login` kept as belt-and-braces (previous goal), but
  generation no longer needs it at all.

## Validation (all with NO HF_TOKEN in env)

- Offline pipeline load from repo cache: OK (91–99 s).
- Full TURBO render via `generate_take.py`: take-05, `ok:true`, 117.3 s,
  seed cycled correctly (42 for take #5). Any network attempt would raise
  under `HF_HUB_OFFLINE=1`, so success proves hermetic generation.

## Limits

- The shim is per-snapshot: a future `setup_ideogram.py` refresh that pulls a
  new upstream commit writes the shim into the new snapshot automatically
  (ensure step runs on every prefetch). If upstream ever ships a real
  `tokenizer/config.json`, the ensure step no-ops.
- Shim mapping table is `{'Qwen2Tokenizer': 'qwen2'}` — an unknown future
  tokenizer class fails setup loudly rather than silently.
