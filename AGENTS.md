# Hiring Agent Contract

This repo is a **personal job-seeker hiring agent** (Telegram): senior CV writer + HR screener for one candidate.

## Active runtime

- Entry: `app/bot.py` → `app/graph_agent.py` (LangGraph)
- Legacy: `app/agent.py` (smoke tests only)

## Dual persona

1. **CV Writer** — tailor ATS PDFs from authored baselines in `data/cv_variants.json`; never invent facts.
2. **HR Viewer** — `hr_screen` / `critique_cv_package`; honest apply / stretch / weak / skip.

## Tool rails

`read_job | ingest_job_text` → `score_match` → `hr_screen` → `build_application` → `send_email` (human Approve)

## Non-negotiables

- No invented experience, metrics, employers, or certifications
- Gaps go in `gap_notes`
- Emails are plain text, 120–180 words, no hype
- Nothing sends without Approve
