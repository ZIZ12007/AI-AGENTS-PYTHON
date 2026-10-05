# AI Agent — Minimal Frontend

Minimalistic chat UI for `Project1/MAIN.PY` (ReAct agent: Gemini + DuckDuckGo search + calculator).

## Layout

```text
Project1/
  MAIN.PY            # original CLI agent (unchanged)
  server.py          # FastAPI wrapper exposing POST /api/chat
  requirements.txt   # backend deps
  frontend/
    index.html       # minimal chat UI
    styles.css       # minimal styling
    app.js           # backend call + demo fallback
```

## Run frontend only (instant demo, no key)

Just open `frontend/index.html` in a browser. It works offline in demo mode:
local calculator parsing + canned answers.

## Run with live agent

```powershell
cd AI-AGENTS-PYTHON/Project1
pip install -r requirements.txt
# create .env with:
# GOOGLE_API_KEY=your_key_here
uvicorn server:app --port 8000
```

Then open http://localhost:8000/ (serves the same frontend) or open
`frontend/index.html` and set Backend URL to `http://localhost:8000/api/chat`
via the ⚙ button.

## API

- `GET /api/health` → `{ ok, has_key, model, tools }`
- `POST /api/chat` body `{ "message": "..." }` →
  `{ "reply": "...", "tools_used": [...], "demo": false }`

Verified: `server.py` compiles, `app.js` passes `node --check`.
