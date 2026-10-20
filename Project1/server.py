"""FastAPI backend for the ReAct AI agent (Gemini + DuckDuckGo search + calculator).

Run:
    pip install -r requirements.txt
    # create .env with GOOGLE_API_KEY=...
    uvicorn server:app --reload --port 8000

The frontend in ./frontend/ talks to POST /api/chat and falls back to its own
demo mode when the backend is unreachable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
# Allow "uvicorn Project1.server:app" from the repo root as well.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent_core import (  # noqa: E402  (path setup has to run first)
    MAX_HISTORY_TURNS,
    build_agent,
    build_model,
    build_search_tool,
    get_api_key,
    get_model_chain,
    get_model_name,
    is_model_unavailable_error,
    is_rate_limit_error,
    stream_reply,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("ai-agent")

load_dotenv(dotenv_path=BASE_DIR / ".env")

FRONTEND_DIR = BASE_DIR / "frontend"

# How long a single agent run may take before the request is abandoned.
# A ReAct turn needs two model calls (decide + answer) and the free Gemini
# tier regularly takes 15-30s per call, so the budget has to be generous.
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "120"))
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))

app = FastAPI(title="AI Agent — Minimal Frontend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MissingAPIKeyError(RuntimeError):
    """Raised when the Gemini API key is not configured."""


_agent = None
_agent_error: str | None = None
# One agent per model id: a rate limited model is skipped, the next one in the
# chain gets its own agent.
_agents: dict[str, Any] = {}
_agent_errors: dict[str, str] = {}


def get_agent(model_name: str | None = None):
    """Build (and cache) the ReAct agent for a model id."""
    global _agent, _agent_error

    name = model_name or get_model_name()
    if name in _agents:
        return _agents[name]
    if name in _agent_errors:
        raise RuntimeError(_agent_errors[name])

    if not get_api_key():
        raise MissingAPIKeyError("GOOGLE_API_KEY is not configured")

    try:
        agent = build_agent(model=build_model(model_name=name))
    except Exception as exc:  # invalid model id, missing dependency, ...
        _agent_errors[name] = f"{type(exc).__name__}: {exc}"
        if name == get_model_name():
            _agent_error = _agent_errors[name]
        log.exception("could not build the agent")
        raise

    _agents[name] = agent
    if name == get_model_name():
        _agent = agent
    return agent


class ChatTurn(BaseModel):
    """One earlier conversation turn replayed to the agent."""

    role: str = Field(default="user")
    content: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    # Earlier turns so the agent can answer follow-up questions. Optional: a
    # single-message request works exactly as before.
    history: List[ChatTurn] = Field(default_factory=list)


# Roles the agent understands. Everything else is dropped.
_HISTORY_ROLES = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "agent": "assistant",
}


def normalise_history(history: List[ChatTurn] | None) -> list[tuple[str, str]]:
    """Turn raw history into a bounded list of ``(role, content)`` pairs."""
    cleaned: list[tuple[str, str]] = []
    for turn in history or []:
        role = _HISTORY_ROLES.get((turn.role or "").strip().lower())
        content = (turn.content or "").strip()
        if not role or not content:
            continue
        cleaned.append((role, content[:MAX_MESSAGE_CHARS]))
    return cleaned[-MAX_HISTORY_TURNS:]


@app.get("/api/health")
def health():
    """Lightweight status probe used by Render / the frontend settings panel."""
    search_available = build_search_tool() is not None
    return {
        "ok": True,
        "has_key": bool(get_api_key()),
        "model": get_model_name(),
        "model_chain": get_model_chain(),
        "tools": ["calculator"] + (["duckduckgo_search"] if search_available else []),
        "search_available": search_available,
        "agent_ready": _agent is not None,
        "timeout_seconds": AGENT_TIMEOUT,
        "max_message_chars": MAX_MESSAGE_CHARS,
        "history_turns": MAX_HISTORY_TURNS,
    }


@app.post("/api/chat")
async def chat(body: ChatIn):
    """Run the ReAct loop and return the answer plus the tools that were used."""
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    history = normalise_history(body.history)

    if not get_api_key():
        return {
            "reply": (
                "GOOGLE_API_KEY is missing on the server. Add it to Project1/.env "
                "(or set it in the host's environment) and restart the backend."
            ),
            "tools_used": [],
            "demo": True,
        }

    started = time.perf_counter()
    deadline = started + AGENT_TIMEOUT
    unavailable: list[str] = []

    for model_name in get_model_chain():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return _timeout_response()

        try:
            agent = await run_in_threadpool(get_agent, model_name)
            reply, tools_used = await asyncio.wait_for(
                run_in_threadpool(stream_reply, agent, message, history),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            log.warning("agent run timed out after %ss", AGENT_TIMEOUT)
            return _timeout_response()
        except MissingAPIKeyError:
            return {
                "reply": "GOOGLE_API_KEY is missing on the server.",
                "tools_used": [],
                "demo": True,
            }
        except Exception as exc:
            if is_model_unavailable_error(exc):
                # The free tier allows only a few requests per model per day and
                # a busy model can be overloaded. Move on to the next model
                # instead of failing the request.
                log.warning(
                    "model %s is unavailable (%s), trying the next one",
                    model_name,
                    type(exc).__name__,
                )
                unavailable.append(model_name)
                continue
            # The details stay in the log; the client only gets a safe summary.
            log.exception("agent run failed")
            return {
                "reply": "The agent failed while answering. The server log has the details.",
                "tools_used": [],
                "demo": False,
                "error": True,
            }

        return {
            "reply": reply,
            "tools_used": tools_used,
            "demo": False,
            "model": model_name,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    # No model in the chain could answer.
    log.warning("no model available: %s", ", ".join(unavailable))
    return {
        "reply": (
            "No Gemini model could answer right now: "
            f"{', '.join(unavailable)}. The free tier meters each model "
            "separately (20 requests/day), and a busy model can be overloaded. "
            "The limit resets in a few hours; to use another model now, set "
            "GEMINI_MODEL in the server environment "
            "(for example GEMINI_MODEL=gemini-3.1-flash-lite)."
        ),
        "tools_used": [],
        "demo": False,
        "error": True,
    }


def _timeout_response() -> dict:
    """The reply used when a run outlasts AGENT_TIMEOUT."""
    return {
        "reply": (
            f"The agent took longer than {AGENT_TIMEOUT:.0f}s, so the request was "
            "given up on. Try a shorter question."
        ),
        "tools_used": [],
        "demo": False,
        "error": True,
    }


# --- serve the minimal frontend (must come after the API routes) -------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def api_not_found(path: str):
        """Unknown API routes answer with JSON, not with the static file server."""
        return JSONResponse(
            status_code=404,
            content={"detail": f"Unknown API route: /api/{path}"},
        )

    # The frontend references "./styles.css" and "./app.js" relatively, so the
    # assets have to be reachable from the site root as well.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD")),
    )
