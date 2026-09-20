# Telegram Hiring Agent

A professional **CV writer + HR screener** you talk to in Telegram. It finds
jobs, judges fit the way a recruiter would, writes a tailored ATS CV PDF, drafts
outreach email, critiques the package, and sends only after you approve. Built
to run 24/7 on Railway from a single container.

**What it is:** your personal hiring squad for landing interviews — not an
employer ATS and not a generic chatbot.

---

## Deploy to Railway in ~15 minutes

### 1. Get your three credentials first

| What | Where | Notes |
|---|---|---|
| Telegram bot token | Message **@BotFather** → `/newbot` | Looks like `123456789:AA...` |
| LLM API key | [build.nvidia.com](https://build.nvidia.com) (free) | Or Groq / OpenRouter / OpenAI — any tool-calling model |
| Gmail App Password | myaccount.google.com → Security → 2-Step Verification → App passwords | 16 characters. Your normal password will **not** work |

### 2. Push this folder to GitHub

```bash
cd job-agent
git init
git add .
git commit -m "Telegram career agent"
git branch -M main
git remote add origin https://github.com/<you>/job-agent.git
git push -u origin main
```

`.gitignore` already excludes `.env`, so no secrets go up.

### 3. Create the Railway service

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → pick the repo.
2. Railway detects `Dockerfile` and `railway.json` and starts building. Let it finish.

### 4. Add a Volume (do not skip this)

Railway's filesystem is wiped on every redeploy. Without a volume, your profile
edits, conversation history and application tracker are gone each time you push.

Service → **Variables** tab → **+ New Volume** → mount path: **`/data`**

### 5. Set Variables

Service → **Variables** → **Raw Editor**, paste and edit:

```
TELEGRAM_BOT_TOKEN=123456789:AA...
LLM_API_KEY=nvapi-...
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=meta/llama-3.3-70b-instruct
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=your16charapppassword
SENDER_EMAIL=you@gmail.com
SENDER_NAME=Your Name
DATA_DIR=/data
TZ=Africa/Cairo
```

Railway redeploys automatically.

### 6. Lock it down

Open your bot in Telegram, send `/whoami`, copy the numeric ID, then add one more
variable and redeploy:

```
TELEGRAM_ALLOWED_USER_ID=<your numeric id>
```

Until you do this, anyone who finds the bot can use your CV, your profile and your
Gmail sending quota.

### 7. Verify

Send `/diag` to the bot. Every line should say *set* or *loaded*. Then send a job
description and watch it work.

---

## Talking to it

There are no magic keywords. It is one continuous conversation:

- *"Find AI Engineer jobs in Cairo posted this week"*
- paste a LinkedIn link, or paste the full job description
- *"Review this as HR — apply, stretch, or skip?"*
- *"Build the tailored CV and email"*
- *"Critique the draft before I send"*
- *"Make the email shorter and less formal"*
- *"Add Recovera to my profile — multi-agent BI system, FastAPI + LangChain"*
- *"What did I apply to this week?"*
- *"Remember I won't take anything under 20k EGP"*

| Command | Does |
|---|---|
| `/start` | What it can do |
| `/profile` | Summary of the profile on file |
| `/apps` | Last 15 applications |
| `/stats` | Totals and last-7-days count |
| `/send x@y.com` | Force-send the staged draft to an address |
| `/cancel` | Discard the staged draft |
| `/reset` | Clear the conversation thread (profile and history survive) |
| `/diag` | Config self-check |
| `/whoami` | Your Telegram ID |

---

## What it can actually do

Fourteen tools, chosen by the model turn by turn:

| Tool | Effect |
|---|---|
| `get_profile` | Reads the entire stored profile before writing anything |
| `update_profile` | Permanently edits your profile from chat |
| `search_jobs` | LinkedIn public job feed, with location / remote / recency filters |
| `read_job` | Pulls the full text of a posting URL and any contact emails in it |
| `ingest_job_text` | Loads a pasted JD when LinkedIn blocks the server (or you paste text) |
| `fetch_url` | Reads a company page or a GitHub repo to verify a claim |
| `score_match` | Deterministic 0–100 fit score — math, not the model's opinion |
| `hr_screen` | Recruiter verdict: apply / stretch / weak / skip + must-have gaps + ATS keywords |
| `build_application` | Generates the tailored ATS PDF and stages the email |
| `critique_cv_package` | Hiring-manager pass on the staged CV + email before Approve |
| `send_email` | **Asks permission.** Cannot send on its own |
| `track_application` | Keeps the application history current |
| `list_applications` | Checks history, so it won't apply to the same job twice |
| `remember` | Stores standing preferences, injected into every future chat |

Pipeline (enforced in code):

`read_job | ingest_job_text` → `score_match` → `hr_screen` → `build_application` → `send_email`

Guardrails that stay on: the system prompt forbids inventing experience, projects,
metrics or certifications; baselines in `cv_variants.json` are fact-locked;
`build_application` requires `gap_notes`; and nothing leaves without your Approve tap.

---

## Architecture

```
Telegram  ──►  bot.py              chat surface, approval buttons, commands
                  │
                  ▼
               graph_agent.py      LangGraph ReAct + tool rails (active)
                  │
     ┌────────────┼────────────┬──────────────┬───────────────┬────────────┐
     ▼            ▼            ▼              ▼               ▼            ▼
  store.py     jobs.py    matching.py   hr_review.py   cv_generator  email_sender
  JSON on      LinkedIn   fit score     HR screen      fpdf2 PDF     Gmail API
  the volume   over HTTP  0–100         apply/skip

health.py binds $PORT so Railway sees a live service.
```

`agent.py` is the legacy hand-rolled loop (kept for offline smoke tests). Production
uses `graph_agent.py`.

### Three things that changed to make this deployable

**Playwright is gone from the hot path.** The previous version drove a Chromium
profile you had logged into by hand. On Railway there is no such profile, no way to
solve a CAPTCHA, and LinkedIn authwalls datacenter IPs — it would have looked
deployed and failed on the first link. `jobs.py` uses LinkedIn's logged-out guest
endpoints over plain HTTP instead, and says so plainly when LinkedIn throttles,
rather than pretending it read the page. The browser scraper survives as
`app/linkedin_browser.py` for local use behind `ENABLE_PLAYWRIGHT=true`.

**State is on disk, not in module-level dicts.** Drafts and paused conversations
used to live in memory. Railway recycles the container on every deploy, crash and
sleep, so the agent would forget a half-finished conversation mid-sentence.
Everything now writes through to JSON on the volume with atomic replace.

**The agent is conversational.** Previously every message was treated as a new job
posting, which made a back-and-forth impossible. Now one persistent thread per user
handles anything you send, and the model replying with plain text *is* how it asks
you a question — no special pause/resume machinery needed.

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # then fill it in
python app/bot.py
```

Before any deploy:

```bash
python tests/smoke_test.py
```

That runs the whole agent loop, the PDF generator, the store and the approval
interrupt against a stubbed LLM — no keys, no network. If it passes, only
credentials and third-party availability can still break you.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Bot silent, Railway logs fine | `TELEGRAM_ALLOWED_USER_ID` doesn't match your ID. Send `/whoami` |
| "The AI provider rejected the request" | Key wrong, or the model doesn't support tool calling. Try `meta/llama-3.3-70b-instruct` |
| Everything resets on redeploy | No volume mounted at `/data`, or `DATA_DIR` isn't `/data` |
| LinkedIn links fail, pasted text works | LinkedIn throttles cloud IPs. Expected — paste the description |
| `❌ Not sent — SMTP login was rejected` | Gmail needs a 16-char App Password, not your account password |
| Two replies to every message | Two deployments running. Keep `numReplicas: 1`, delete old services |

**Cost:** Railway's free tier covers this (~200 MB image, idles near zero CPU).
NVIDIA NIM's free tier covers normal job-hunting volume. Gmail SMTP allows about
500 sends/day.
