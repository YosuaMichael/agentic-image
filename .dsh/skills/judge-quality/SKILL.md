---
name: judge-quality
description: >
  Review rendered takes for a session: caption-alignment check, artifact
  flags, and a ranked verdict in review.json. No LLM-as-judge scoring —
  deterministic flags plus the model's own verifier notes. Use only after the
  user explicitly asks for a review.
---

# Skill: judge-quality

## Policy

Deterministic checks only, plus human eyes. There is no CLAP/aesthetic-score
gating in this repo (no scorer is wired); the agent reviews takes against the
brief + caption and records flags. Never present an unverified opinion as a
metric.

## Procedure

1. **Learnings check.** Skim `studio/learnings/LEARNINGS.md` first.
2. For each take in `studio/sessions/<id>/takes/`:
   - Open its result link (`/view/...`) and LOOK at the image (read the PNG
     with image tooling when available; otherwise open the link for the user
     and review from their feedback).
   - Check alignment against the take's FROZEN `take-NN.caption.json`
     (not the session root — the root may have moved on): subject present?
     text literals rendered verbatim? palette/composition honored?
   - Check `take-NN.metadata.json`: `warnings` (caption-verifier notes),
     `actual_width/actual_height` vs requested, `elapsed_s` anomalies,
     `dry_run` (placeholders are not reviewable takes — say so).
   - Flag defects: `text-mismatch` (rendered text ≠ literal), `missing-subject`,
     `wrong-size`, `safety-block` (gray filter frame), `artifact` (visible
     garbage anatomy/geometry), `off-palette`, `off-brief`.
3. Write `review.json` (`review/v1`):
   ```json
   {"schema": "review/v1", "session": "<id>",
    "takes": [{"take": "take-01", "seed": 7,
               "prompt_scored": "takes/take-01.caption.json",
               "flags": ["text-mismatch"], "notes": "..."}],
    "ranking": ["take-02", "take-01"], "recommended": "take-02",
    "notes": "..."}
   ```
   Rank = fewer flags first, then closer alignment to the brief, then user
   preference when they weighed in. Recommend exactly one.
4. Present a table `take | seed | size | flags | notes`, the recommendation,
   and offer: love-it (done) / adjust caption (compose-brief loop, `_vN`
   versioning on superseded artifacts — never overwrite a take's frozen
   copies) / render more takes (generate-image) / start over.

## Handoff

Review is advisory: the user picks the winner. Record their pick in
`review.json` (`"picked": "<take>"`) when they state it.
