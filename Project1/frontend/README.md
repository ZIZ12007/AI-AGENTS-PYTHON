# AI Agent — Minimal Frontend

Minimalistic chat UI for the ReAct agent (Gemini + DuckDuckGo search + calculator).

## Layout

```text
Project1/
  MAIN.PY            # CLI agent
  agent_core.py      # shared model + tools + streaming helpers
  server.py          # FastAPI wrapper exposing POST /api/chat
  requirements.txt   # backend deps
  test_agent.py      # offline smoke test
  frontend/
    index.html       # minimal chat UI
    styles.css       # minimal styling
    app.js           # backend call + demo fallback
```

## Run frontend only (instant demo, no key)

Just open `frontend/index.html` in a browser. It works offline in demo mode:
local calculator parsing + canned answers.

## Run with live agent

```bash
cd Project1
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# .env with: GOOGLE_API_KEY=your_key_here
uvicorn server:app --port 8000
```

Then open http://localhost:8000/ — the backend serves this same frontend, so
the UI talks to the API on its own origin and the header shows **connected**.

Opened straight from disk (`frontend/index.html`) the page falls back to
`http://localhost:8000/api/chat`, which is the local backend.

### Backend URL

The ⚙ button sets where the API lives. All of these work:

| Value | Result |
| --- | --- |
| `/api/chat` | the default — the backend that served this page |
| `https://your-service.onrender.com` | a base URL; `/api/chat` is appended for you |
| `http://localhost:8000/api/chat` | an explicit URL |

**Reset** goes back to the same-origin default. The choice is stored in
`localStorage`, so it survives a reload.

## API

- `GET /api/health` → `{ ok, has_key, model, tools, search_available, agent_ready, timeout_seconds, max_message_chars, history_turns }`
- `POST /api/chat` body `{ "message": "...", "history": [...] }` →
  `{ "reply": "...", "tools_used": [...], "demo": false, "elapsed_ms": 913 }`
  (`demo: true` when the server has no API key, `error: true` when the agent failed)

`history` is optional and holds the earlier turns as
`{ "role": "user" | "assistant", "content": "..." }`. The UI sends the last 12
turns so follow-up questions keep the context of the conversation.

The UI only fakes a demo reply when the backend cannot be reached at all. An
HTTP error (500, 422, …) is shown with the server's own message.
