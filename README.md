# AI-AGENTS-PYTHON — ReAct AI Agent + Minimal Chat Frontend

A Python ReAct agent (Gemini + DuckDuckGo search + calculator) with a clean,
minimal chat UI and a FastAPI backend. Open the frontend for an instant demo —
no API key needed. Connect the backend for live search + LLM answers.

![desktop](docs/screenshots/desktop.png)

> Live demo (frontend-only): enable GitHub Pages (docs below) →
> `https://ZIZ12007.github.io/AI-AGENTS-PYTHON/`

## What it does

- Chat UI sends a message to `POST /api/chat` and keeps the conversation history
- Backend runs a ReAct loop: Thought → tool → Observation → answer
- Tools: `duckduckgo_search` (live web), `calculator` (local arithmetic)
- UI shows which tools fired per reply, and reports backend errors instead of hiding them
- The backend serves the frontend, so the UI talks to the API on its own origin by default

## Repo map

```text
Project1/
  MAIN.PY              # CLI ReAct agent
  agent_core.py        # shared model + tools + streaming helpers
  server.py            # FastAPI wrapper (POST /api/chat, GET /api/health, serves frontend)
  requirements.txt     # backend deps
  test_agent.py        # offline smoke test (no API key needed)
  .env.example         # copy to .env and add your key
  frontend/
    index.html / styles.css / app.js   # minimal chat (demo fallback built in)
docs/
  screenshots/desktop.png / mobile.png
  PORTFOLIO.md         # copy-paste portfolio write-up
  LOOM-SCRIPT.md       # 60-second demo script
DEPLOY.md              # every hosting option (Vercel, Cloud Run, tunnel, ...)
```

## Run it

Frontend only (instant):

```bash
# just open in a browser
Project1/frontend/index.html
```

Full stack:

```bash
cd Project1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# .env -> GOOGLE_API_KEY=your_key_here  (free key: https://aistudio.google.com/apikey)
uvicorn server:app --port 8000
# open http://localhost:8000/
```

API:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is 24 + 58?"}'
```

```json
{ "reply": "The sum of 24 and 58 is 82.", "tools_used": ["calculator"], "demo": false, "elapsed_ms": 913 }
```

Follow-up questions keep the context of the conversation — send the earlier
turns along with the new message (the last 12 turns are replayed):

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
        "message": "And what tools does it have?",
        "history": [
          {"role": "user", "content": "What is a ReAct agent?"},
          {"role": "assistant", "content": "It loops thought/tool/observation."}
        ]
      }'
```

Health check:

```bash
curl http://localhost:8000/api/health
# {"ok":true,"has_key":true,"model":"gemini-3.8-flash","tools":["calculator","duckduckgo_search"], ...}
```

| Response field | Meaning |
| --- | --- |
| `reply` | the answer (or the reason the agent could not answer) |
| `tools_used` | the tools that fired, e.g. `["calculator"]` |
| `demo` | `true` when the server has no API key, `false` when the agent ran |
| `error` | `true` when the agent failed or timed out |
| `model` | the model that answered (see the fallback chain below) |
| `elapsed_ms` | how long the run took |

### Model fallback

The Gemini free tier meters **each model separately** (20 requests/day) and
busy models answer `503 high demand`. So the server keeps a chain of models
and moves to the next one when the current one is rate limited or overloaded:

```
gemini-3.8-flash → gemini-3.7-flash → gemini-3.6-flash → gemini-3.5-flash → gemini-3.1-flash-lite
```

`GEMINI_MODEL` puts your own model first; the chain is reported by
`/api/health` as `model_chain`. A quota error fails in about two seconds
(the SDK's retry storm is disabled) so the fallback is cheap.

> The free tier is generous enough for a demo but not for real traffic —
> enable billing in AI Studio, or expect the "no quota left" message after
> ~20 requests per model per day.

Offline smoke test (no API key required):

```bash
cd Project1 && python test_agent.py
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | – | Gemini API key (required for live answers) |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Model id used by the agent |
| `AGENT_TIMEOUT` | `120` | Seconds a single agent run may take |
| `PORT` | `8000` | Port used by `python server.py` |

> The free Gemini tier regularly takes 15–30s per model call, and a ReAct turn
> needs two of them (decide → answer). Keep `AGENT_TIMEOUT` comfortably above
> 60, or slow answers get cut off.

## Deploy a live site (pick one)

See **[DEPLOY.md](./DEPLOY.md)** for the full guide — Vercel (recommended,
free, no card), a Cloudflare Tunnel for a live URL straight from your PC,
Google Cloud Run, Render, or the frontend-only GitHub Pages demo.

**A. Frontend demo on GitHub Pages (free, 2 min)** — Settings → Pages → GitHub Actions, the workflow in `.github/workflows/pages.yml` publishes `Project1/frontend`. No key needed (demo mode). Point it at a hosted backend by setting one line in `Project1/frontend/config.js`.

**B. Full backend on Vercel (free, no card)** — `vercel.json` is ready. Import the repo, set `GOOGLE_API_KEY`, deploy. The API and the UI share one domain.

**C. Full backend on Render (free)** — `render.yaml` included:

```bash
# Render dashboard -> New -> Blueprint -> select this repo
```

Set env `GOOGLE_API_KEY`. The backend also serves the frontend, so
`https://<your-service>.onrender.com/` is a complete app — no extra
configuration needed.

**D. Vercel (frontend)** — `vercel.json` included:

```bash
npx vercel --prod
```

## Screenshots

| Desktop | Mobile |
| --- | --- |
| ![](docs/screenshots/desktop.png) | ![](docs/screenshots/mobile.png) |

## Stack

Python · LangChain v1 (`create_agent`) · LangGraph · Gemini (`gemini-3.8-flash`) · DuckDuckGo (`ddgs`) · FastAPI · Vanilla JS · GitHub Pages / Render

## Architecture

```mermaid
flowchart LR
  U[Minimal chat UI] -->|POST /api/chat| API[FastAPI server.py]
  API --> A[ReAct agent]
  A -->|tool| S[DuckDuckGo search]
  A -->|tool| C[Calculator]
  A --> G[Gemini 3.8 Flash]
  G --> A --> API --> U
  U -.->|backend offline| D[Local demo fallback]
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: Could not import ddgs python package` | `pip install -r requirements.txt` (the `ddgs` package provides the search backend) |
| `400 INVALID_ARGUMENT ... API key not valid` | wrong/missing key — set `GOOGLE_API_KEY` in `Project1/.env` |
| `404 models/gemini-2.0-flash is not found` | `gemini-2.0-flash` was shut down; the default is now `gemini-3.8-flash` |
| `GOOGLE_API_KEY is missing on the server` | the API returns `demo: true`; add the key and restart the server |
| Frontend loads but the page is unstyled | open the backend root (`http://localhost:8000/`) instead of `index.html` — the API serves `styles.css` / `app.js` |
| The header says "backend offline" on a deployed site | the Backend URL in ⚙ still points at `http://localhost:8000`. Press **Reset** to go back to the same-origin default (`/api/chat`) |
| The UI answers with a canned demo reply | the backend is unreachable. The error bubble above it names the URL that failed |
| A reply says "The agent failed while answering" | the server log has the traceback; the API only returns a safe summary |
| "The agent took longer than Ns" | the free Gemini tier is slow or rate-limited. Raise `AGENT_TIMEOUT` and retry |
