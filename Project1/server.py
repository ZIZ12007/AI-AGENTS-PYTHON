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
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
# Allow "uvicorn Project1.server:app" from the repo root as well.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent_core import (  # noqa: E402  (path setup has to run first)
    MAX_HISTORY_TURNS,
    PROVIDERS,
    PROVIDER_ORDER,
    available_providers,
    available_models,
    build_agent,
    build_model,
    build_search_tool,
    get_api_key,
    get_model_chain,
    get_model_name,
    get_provider,
    is_model_unavailable_error,
    is_rate_limit_error,
    model_is_allowed,
    model_latencies,
    order_models,
    record_model_latency,
    stream_reply,
    stream_reply_events,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("ai-agent")

load_dotenv(dotenv_path=BASE_DIR / ".env")

FRONTEND_DIR = BASE_DIR / "frontend"

# How long a single agent run may take before the request is abandoned.
# A ReAct turn needs two model calls (decide + answer) and the free Gemini
# tier regularly takes 15-30s per call, so the budget has to be generous.
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "120"))
# How long one model may take before it is written off and the next one is
# tried. Keeps a single slow model from eating the whole request.
MODEL_ATTEMPT_SECONDS = float(os.getenv("MODEL_ATTEMPT_SECONDS", "15"))
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
    # The model the client wants to use. It must be one of the models the
    # server's keys can reach; anything else is ignored.
    model: str | None = None


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
    provider = get_provider()
    spec = PROVIDERS.get(provider) or {}
    return {
        "ok": True,
        "has_key": bool(get_api_key()),
        "provider": provider,
        "provider_label": spec.get("label", provider),
        "providers_available": available_providers(),
        "model": get_model_name(),
        "model_chain": get_model_chain(),
        "models_available": available_models(),
        "model_latency_ms": {k: int(v * 1000) for k, v in model_latencies().items()},
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
            "reply": _no_key_message(),
            "tools_used": [],
            "demo": True,
        }

    # A repeated question is answered from the cache: no API call, no waiting.
    cache_key = _cache_key(message, history)
    cached = cache_get(cache_key)
    if cached is not None:
        answer = dict(cached)
        answer["cached"] = True
        answer["elapsed_ms"] = 0
        return answer

    started = time.perf_counter()
    deadline = started + AGENT_TIMEOUT
    unavailable: list[str] = []

    # Fastest known models first, slow ones parked at the end.
    candidates = order_models(get_model_chain())

    # A model the client asked for goes first, so the picker in the UI is
    # honoured. Unknown models are ignored rather than sent to the provider.
    if body.model and model_is_allowed(body.model):
        candidates = [body.model] + [m for m in candidates if m != body.model]

    for index, model_name in enumerate(candidates):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return _timeout_response()

        # Every model but the last gets a short leash: a model that cannot
        # answer within it is parked and the next one takes over. The last one
        # may use whatever is left.
        last = index == len(candidates) - 1
        attempt_budget = remaining if last else min(remaining, MODEL_ATTEMPT_SECONDS)

        attempt_started = time.perf_counter()
        try:
            agent = await run_in_threadpool(get_agent, model_name)
            reply, tools_used = await asyncio.wait_for(
                run_in_threadpool(stream_reply, agent, message, history),
                timeout=attempt_budget,
            )
        except asyncio.TimeoutError:
            took = time.perf_counter() - attempt_started
            if not last:
                log.warning(
                    "model %s did not answer within %.0fs, trying the next one",
                    model_name,
                    attempt_budget,
                )
                record_model_latency(model_name, took)
                unavailable.append(f"{model_name} (too slow)")
                continue
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
                "reply": _safe_error_message(exc),
                "tools_used": [],
                "demo": False,
                "error": True,
            }

        took = time.perf_counter() - attempt_started
        record_model_latency(model_name, took)
        log.info("model %s answered in %.1fs", model_name, took)
        answer = {
            "reply": reply,
            "tools_used": tools_used,
            "demo": False,
            "model": model_name,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        cache_set(cache_key, answer)
        return answer

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


def _no_key_message() -> str:
    """What to say when no provider key is configured."""
    options = "\n".join(
        f"  - {PROVIDERS[name]['label']}: set {PROVIDERS[name]['env'][0]} "
        f"({PROVIDERS[name]['url']})"
        for name in PROVIDER_ORDER
    )
    return (
        "No AI provider key is configured on the server. Add one of these to "
        "Project1/.env (or to the host's environment) and restart:\n"
        f"{options}\n"
        "The fastest free option is Groq or OpenRouter."
    )


def _safe_error_message(exc: BaseException) -> str:
    """A message that is safe to show the client.

    Provider exceptions carry the API endpoint, the quota details and the
    model id - none of which belong in a chat bubble.
    """
    if is_rate_limit_error(exc):
        return (
            "The AI provider's free quota is used up for now. It resets later "
            "today; a different provider or an enabled billing plan avoids the wait."
        )
    if is_model_unavailable_error(exc):
        return (
            "The AI provider is temporarily overloaded and every model in the "
            "chain is busy. Try again in a moment."
        )
    return "The agent failed while answering. The server log has the details."


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


# --------------------------------------------------------------------------- #
# Answer cache
# --------------------------------------------------------------------------- #
# The free tier allows 20 requests per model per day and a ReAct turn costs
# two of them, so asking the same question again should not spend another
# one. The cache also makes repeated questions instant.
_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL = float(os.getenv("CACHE_TTL", "3600"))
CACHE_MAX = int(os.getenv("CACHE_MAX", "200"))


def _cache_key(message: str, history: list[tuple[str, str]]) -> str:
    payload = json.dumps([message, history], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_get(key: str) -> dict | None:
    """A cached answer, if it is still fresh."""
    entry = _CACHE.get(key)
    if entry is None:
        return None
    stored_at, data = entry
    if time.time() - stored_at > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return data


def cache_set(key: str, data: dict) -> None:
    """Remember an answer, evicting the oldest one when the cache is full."""
    if len(_CACHE) >= CACHE_MAX:
        oldest = min(_CACHE.items(), key=lambda item: item[1][0])[0]
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.time(), data)


# --- streaming endpoint ----------------------------------------------------
def _sse(event: str, data: dict) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(body: ChatIn):
    """Stream the answer as it is written, so the UI does not sit silent.

    Events: ``delta`` (a piece of the answer), ``done`` (the finished reply
    with the tools it used) and ``error``.
    """
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    history = normalise_history(body.history)

    if not get_api_key():
        async def demo_stream():
            yield _sse(
                "done",
                {
                    "reply": _no_key_message(),
                    "tools_used": [],
                    "demo": True,
                },
            )

        return StreamingResponse(demo_stream(), media_type="text/event-stream")

    cache_key = _cache_key(message, history)
    cached = cache_get(cache_key)
    if cached is not None:
        async def cached_stream():
            yield _sse("delta", {"text": cached["reply"]})
            yield _sse(
                "done",
                {
                    "reply": cached["reply"],
                    "tools_used": cached.get("tools_used", []),
                    "model": cached.get("model"),
                    "cached": True,
                    "elapsed_ms": 0,
                },
            )

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        """Run the agent in a worker thread and push its output into the queue."""
        try:
            for kind, *rest in stream_reply_events(agent, message, history):
                loop.call_soon_threadsafe(queue.put_nowait, (kind, rest))
        except Exception as exc:  # the client gets a safe error event
            # Never send the raw exception: it carries provider URLs, quota
            # details and the API key. The details stay in the log.
            log.exception("streaming agent run failed")
            safe_message = _safe_error_message(exc)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", [safe_message]))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("end", []))

    async def event_stream():
        task = asyncio.create_task(asyncio.to_thread(produce))
        started = time.perf_counter()
        reply = ""
        tools: list[str] = []
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "delta":
                    reply += payload[0]
                    yield _sse("delta", {"text": payload[0]})
                elif kind == "done":
                    reply, tools = payload[0], payload[1]
                    yield _sse(
                        "done",
                        {
                            "reply": reply,
                            "tools_used": tools,
                            "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        },
                    )
                elif kind == "error":
                    yield _sse("error", {"message": payload[0]})
                elif kind == "end":
                    break
        finally:
            await task

    # The agent for the fastest known model is built before the response
    # starts, so a slow import does not delay the first token.
    agent = await run_in_threadpool(get_agent, order_models(get_model_chain())[0])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
