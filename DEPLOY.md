# Deploying AI-AGENTS-PYTHON

The backend serves the frontend, so the whole app deploys as **one service**
and the UI talks to the API on its own origin — no CORS, no extra config.

| Option | Cost | Card? | Best for |
| --- | --- | --- | --- |
| **A. Vercel** | Free | No | The easiest full-stack deploy (recommended) |
| **B. Cloudflare Tunnel** | Free | No | A live URL *right now*, straight from your PC |
| **C. Google Cloud Run** | Free tier | Yes | A container that scales and sleeps to zero |
| **D. Render** | Free | Yes | The `render.yaml` blueprint in this repo |
| **E. GitHub Pages** | Free | No | The frontend-only demo (no live AI) |

---

## A. Vercel (recommended)

Vercel's Python runtime runs FastAPI natively, and the Hobby plan allows
**300-second functions** — long enough for a ReAct turn (the free Gemini tier
often takes 15–30s per model call).

`vercel.json` is already set up: the static frontend and the Python function
are built separately and routed onto one domain.

1. Go to https://vercel.com/new → **Import** the repo `ZIZ12007/AI-AGENTS-PYTHON`
2. Framework preset: **Other** (the `vercel.json` drives the build)
3. Add the environment variable `GOOGLE_API_KEY` (and optionally `GEMINI_MODEL`)
4. Deploy

The API is then at `https://<your-app>.vercel.app/api/chat` and the UI at
`https://<your-app>.vercel.app/`.

> **If the build fails**, set the project's **Root Directory** to `Project1`.
> Vercel then auto-detects `server.py` and `requirements.txt` with no config
> at all, and FastAPI serves the frontend from its own bundle.

## B. Cloudflare Tunnel (live in 30 seconds, no account)

Exposes the app running on your own machine through a public HTTPS URL. The
URL changes every time you restart the tunnel, and the app only stays online
while your PC is on — so treat it as a demo link, not a deployment.

```bash
# download cloudflared, then:
cloudflared tunnel --url http://127.0.0.1:8000
```

It prints a `https://<random>.trycloudflare.com` URL that serves the full app.
For a permanent URL, create a **named tunnel** (still free) and point your own
domain at it.

## C. Google Cloud Run

The `Dockerfile` in this repo is already Cloud Run ready: it listens on
`$PORT` and has a health check on `/api/health`.

```bash
gcloud run deploy ai-agent --source . --region europe-west1 \
  --set-env-vars GOOGLE_API_KEY=your_key_here
```

Free tier: 2M requests/month, and it scales to zero between requests. A card
is required for the account, but the free tier is not charged.

## D. Render

`render.yaml` is a ready-to-use blueprint:

```bash
# Render dashboard -> New -> Blueprint -> select this repo
```

Set `GOOGLE_API_KEY`. The backend also serves the frontend, so
`https://<your-service>.onrender.com/` is the complete app.

> Render needs a payment method on file before it builds anything
> (`402 Payment information is required`). The free web service is still $0 —
> the card only verifies the account.

## E. GitHub Pages (frontend only)

`.github/workflows/pages.yml` publishes `Project1/frontend` on every push to
`main`, so the demo lives at
`https://ZIZ12007.github.io/AI-AGENTS-PYTHON/`.

It is **frontend only** — there is no API behind it, so the UI runs in demo
mode. To point it at a hosted backend, edit one line in
`Project1/frontend/config.js`:

```js
window.AGENT_CONFIG = { backendUrl: "https://your-app.vercel.app" };
```

A base URL is enough; the `/api/chat` path is added automatically. The UI
shows **backend offline** when the API cannot be reached, instead of
pretending everything is fine.

---

## What to set in the environment

| Variable | Required | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | yes | The Gemini key. Without it the API returns `demo: true` |
| `GEMINI_MODEL` | no | Model id (default `gemini-3.5-flash`, the fastest measured) |
| `AGENT_TIMEOUT` | no | Seconds per agent run (default `120`) |
| `MODEL_ATTEMPT_SECONDS` | no | Seconds one model may take before the next is tried (default `15`) |
| `SLOW_MODEL_SECONDS` | no | A model slower than this is parked (default `8`) |
| `CACHE_TTL` | no | How long an answer stays cached (default `3600`) |
| `PORT` | no | Set by the host |

## Before you deploy

```bash
cd Project1
python test_agent.py        # 44 offline checks, no API key needed
```

## Notes on the free Gemini tier

The free tier allows **20 requests per day per model**, and busy models answer
`503 high demand`. The server handles both: it fails a quota error in about
two seconds and moves on to the next model in the chain
(`gemini-3.8-flash → 3.7 → 3.6 → 3.5 → 3.1-flash-lite`).

For real traffic, enable billing in AI Studio.
