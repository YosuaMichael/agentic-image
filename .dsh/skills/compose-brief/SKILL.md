---
name: compose-brief
description: >
  Turn a user's image idea into the full pre-generation artifact set:
  brief.md (interview result), prompt.txt (plain-language working prompt) and
  caption.json (Ideogram 4 structured JSON caption). Includes the
  user-interaction protocol. Use at the start of every image session.
origin: >
  Prompting rules follow Ideogram's open-source prompting guide
  (oss/ideogram4/docs/prompting.md); the interview/preview-and-confirm
  protocol mirrors agentic-music's compose-brief.
---

# Skill: compose-brief

## Inputs

- The user's request: anything from a one-liner ("a cozy cafe logo") to a
  detailed spec. Everything Ideogram 4 can render is in scope: photorealism,
  illustration, typography/poster design, product shots, wallpapers, banners.

## Outputs (create `studio/sessions/<image-id>/`, id = `YYYYMMDD-HHMMSS-<slug>`)

| Artifact | Content |
|---|---|
| `brief.md` | Full interview result: subject, style/medium, mood, composition, colors, text-to-render, size/aspect, references, exclusions |
| `prompt.txt` | Plain-language working prompt (1–4 sentences). Feedstock for `--use-magic-prompt`, and the human-readable record of intent |
| `caption.json` | **The artifact the model reads**: structured JSON caption — `high_level_description` + `style_description` + `compositional_deconstruction` (required). Written by YOU per the schema below, then gated by `scripts/verify_caption.py` |

There is exactly ONE image model, so there is **no model question and no
`model.json`**. Go straight to the interview.

## Step 0 — Learnings check

Skim `studio/learnings/LEARNINGS.md` first (gitignored; if missing, treat as
empty); apply its rules throughout the interview and artifact writing.

## Step 1 — Interview

Minimum to elicit before writing `brief.md`:

- **Subject**: what/whom is in the image (be concrete: "a ginger cat in a tiny
  wizard hat", not "something cute")
- **Medium**: `photograph` (then ask camera/lens feel) vs illustration /
  3D render / painting / graphic_design (then ask art style)
- **Composition**: orientation + aspect (square 1:1, portrait 2:3, landscape
  3:2, banner …), background, key elements and their placement
- **Style**: aesthetics keywords, lighting, color palette (ask for 3–6 hex
  colors when the user cares about the scheme)
- **Text in image** (Ideogram 4's headline capability): exact literal strings
  to render, and roughly where. Spell them out — the model renders them
  verbatim, typos included
- **Size**: default 1024×1024; offer alternatives from the model-guide skill
  (portrait 1024×1536, landscape 1536×1024, wallpaper 1024×1792 …). Both sides
  must be multiples of 16, 256–2048, aspect ≤ 6:1
- **Exclusions**: what must NOT appear (record verbatim; JSON captions have no
  negative-prompt field, so exclusions steer the `desc` wording instead)

Mode: **Basic** (clear one-liner → infer everything, confirm once) or
**Advanced** (user wants control over caption JSON / layout / palette).
Ambiguity is resolved by asking, never by assuming silently.

## Step 2 — caption.json (the structured caption)

Write the JSON **yourself** — do not rely on magic prompt for the default
path. (Magic prompt is the fallback for casual one-liners, enabled per take
with `--use-magic-prompt`; the hand-written caption is higher quality and
works offline with no API key.) Schema (full reference:
`oss/ideogram4/docs/prompting.md`):

```json
{
  "high_level_description": "One or two sentences summarising the whole image.",
  "style_description": {
    "aesthetics": "moody, cinematic",
    "lighting": "low-key, deep shadows",
    "photo": "35mm, f/1.4",
    "medium": "photograph",
    "color_palette": ["#1B1B2F", "#E43F5A"]
  },
  "compositional_deconstruction": {
    "background": "Required: the environment.",
    "elements": [
      {"type": "obj", "desc": "Required per element; bbox optional."},
      {"type": "text", "text": "LITERAL", "desc": "What the text looks like."}
    ]
  }
}
```

Hard rules (the model was trained on exactly this shape):

- Top level: `high_level_description` (strongly recommended), then
  `style_description`, then **required** `compositional_deconstruction`
  (with required `background` first, then required `elements`).
- `style_description` holds **exactly one** of `photo` (photographs, with
  `medium: "photograph"`) / `art_style` (everything else). Key order is
  strict: photo → `aesthetics, lighting, photo, medium, color_palette`;
  non-photo → `aesthetics, lighting, medium, art_style, color_palette`
  (`color_palette` optional but last when present).
- Elements: `obj` → `type, bbox, desc, color_palette`; `text` →
  `type, bbox, text, desc, color_palette`. `bbox` is optional
  `[y_min, x_min, y_max, x_max]` in 0–1000; `color_palette` ≤ 5 per element,
  ≤ 16 globally; hex MUST be uppercase `#RRGGBB`.
- Serialize compact (`separators=(",", ":")`, `ensure_ascii=False`).
- **Plain-text prompts must NEVER go to the model directly** — they degrade
  quality and trip safety false-positives. Either write caption.json (this
  step) or use magic prompt (`--use-magic-prompt`).

## Step 3 — Verify, then preview-and-confirm (mandatory)

1. Run `python scripts/verify_caption.py --caption studio/sessions/<id>/caption.json`.
   Fix every `error` before proceeding (warnings are judgement calls).
2. Show the user: one-paragraph creative summary, size/aspect, the
   `high_level_description`, text-to-render literals, palette, and the verify
   result. Generate nothing until the user confirms or requests edits. Loop
   edits through Steps 1–2 as needed.
3. Before the gate, list the session folder and confirm artifacts on disk:
   `brief.md`, `prompt.txt`, `caption.json` all exist **non-empty**. A merged
   or dropped tool invocation can silently lose a write — verify on disk
   instead of trusting earlier tool results.

## Handoff

On confirmation, tell the user you are invoking `generate-image` with this
session folder. Mention the sampler default (V4_TURBO_12, fast ~12-step drafts;
request V4_QUALITY_48 explicitly for finals) and
that the first take is 1 image unless they asked for more.
