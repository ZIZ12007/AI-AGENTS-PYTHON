"""Shared building blocks for the ReAct agent (model + tools + streaming).

Used by both entry points:

* ``server.py``  - FastAPI backend (``POST /api/chat``)
* ``MAIN.PY``    - terminal chat loop

Everything here is deliberately import-safe: no heavy model is created at
import time, so the API can boot (and report ``/api/health``) even when the
API key or an optional dependency is missing.
"""
from __future__ import annotations

import ast
import operator
import os
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Gemini 2.0 Flash was shut down on 2026-06-01, so it can no longer be used.
# Override with GEMINI_MODEL=... when a different model is preferred.
DEFAULT_MODEL = "gemini-3.5-flash"

# The Gemini free tier enforces a *per-model* daily quota (20 requests/day
# for the flash models), so one exhausted model should not take the whole app
# down. These are tried in order when the primary is rate limited or busy.
FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
)

# Measured on the free tier: a full ReAct turn (two model calls) takes
# 2.5-4s on gemini-3.5-flash, while other models regularly take 30-100s.
# "lite" is not faster here - it is often slower.
PREFERRED_MODEL = "gemini-3.5-flash"

# How many previous turns are replayed to the agent so it keeps the context of
# a conversation. Keep in sync with MAX_HISTORY_TURNS in server.py.
MAX_HISTORY_TURNS = 12

# Read/connect timeout for a single call to the Gemini API. The API has been
# seen to send the response headers and then stall mid-body.
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))

# How much search output the model gets to read. Shorter snippets mean fewer
# tokens and a faster second model call.
SEARCH_RESULTS = int(os.getenv("SEARCH_RESULTS", "4"))
SEARCH_SNIPPET_CHARS = int(os.getenv("SEARCH_SNIPPET_CHARS", "220"))

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that solves problems with a ReAct loop: "
    "thought -> tool call -> observation -> final answer.\n"
    "Use the 'calculator' tool for any arithmetic (never compute it yourself) and "
    "the 'duckduckgo_search' tool for facts about the world, current events, or "
    "anything you are not certain about.\n"
    "Keep the final answer concise, and mention the sources you used when you "
    "rely on a web search."
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    """Return the Gemini API key from the environment (or ``None``)."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return key.strip() if key and key.strip() else None


def get_model_name() -> str:
    """Return the Gemini model id to use."""
    return (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_model_chain() -> list[str]:
    """The models to try, in order: the configured one first, then fallbacks.

    The measured-fastest model is put in front of the fallbacks so a normal
    request does not have to walk past the slow ones.
    """
    chain = [get_model_name()]
    for name in (PREFERRED_MODEL,) + FALLBACK_MODELS:
        if name not in chain:
            chain.append(name)
    return chain


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when the provider refused the call because of a quota limit."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    return (
        "RESOURCE_EXHAUSTED" in text
        or "429" in text
        or "rate limit" in lowered
        or "quota" in lowered
    )


def is_model_unavailable_error(exc: BaseException) -> bool:
    """True when another model could still answer.

    That is the case for a quota limit (the free tier meters each model
    separately) and for a temporary overload ("this model is experiencing high
    demand"), which usually clears up on the next model.
    """
    if is_rate_limit_error(exc):
        return True
    lowered = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in lowered
        for marker in (
            "unavailable",
            "overloaded",
            "high demand",
            "internal error",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
        )
    )


def model_latencies() -> dict[str, float]:
    """The measured latency per model, for the health endpoint."""
    return dict(_MODEL_LATENCY)


# --------------------------------------------------------------------------- #
# Model performance memory
# --------------------------------------------------------------------------- #
# The free tier is wildly inconsistent: the same model can answer in 2.5s and
# in 36s. Remembering what was fast lets us skip the slow ones instead of
# paying their latency on every request.
_MODEL_LATENCY: dict[str, float] = {}
_MODEL_SLOW_UNTIL: dict[str, float] = {}

# A model slower than this is put aside for a while.
SLOW_MODEL_SECONDS = float(os.getenv("SLOW_MODEL_SECONDS", "8"))
SLOW_MODEL_COOLDOWN = float(os.getenv("SLOW_MODEL_COOLDOWN", "600"))


def record_model_latency(model: str, seconds: float) -> None:
    """Remember how long a model took, and park it if it was too slow."""
    if not model:
        return
    previous = _MODEL_LATENCY.get(model)
    # Keep a running average so one slow call does not condemn a good model.
    average = seconds if previous is None else (previous * 0.6 + seconds * 0.4)
    _MODEL_LATENCY[model] = average
    if average > SLOW_MODEL_SECONDS:
        _MODEL_SLOW_UNTIL[model] = time.time() + SLOW_MODEL_COOLDOWN


def model_is_slow(model: str) -> bool:
    """True while a model is parked because it was too slow."""
    until = _MODEL_SLOW_UNTIL.get(model)
    if until is None:
        return False
    if time.time() >= until:
        _MODEL_SLOW_UNTIL.pop(model, None)
        return False
    return True


def order_models(chain: Iterable[str]) -> list[str]:
    """Order the chain for speed: known-fast first, slow models last."""
    models = list(dict.fromkeys(chain))
    fresh = [m for m in models if not model_is_slow(m)]
    parked = [m for m in models if model_is_slow(m)]
    # Within the healthy group, the fastest known model goes first.
    fresh.sort(key=lambda m: _MODEL_LATENCY.get(m, 0.0) or SLOW_MODEL_SECONDS)
    return fresh + parked


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Evaluate a whitelisted arithmetic AST (no names, calls or attributes)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def _format_number(value: float) -> str:
    """Render a float without a pointless trailing ``.0``."""
    if isinstance(value, int) or value.is_integer():
        return str(int(value))
    return f"{value:g}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as "24 + 58", "7 * 8" or "(3 + 4) / 2".

    Use this for every calculation instead of doing the maths yourself.
    """
    text = (expression or "").strip()
    if not text:
        return "Error: no expression given."
    try:
        tree = ast.parse(text, mode="eval")
        result = _eval_node(tree)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"Error: could not evaluate {text!r} ({exc})."

    number = _format_number(result)
    parts = text.split()
    if len(parts) == 3 and parts[1] == "+":
        return f"The sum of {parts[0]} and {parts[2]} is {number}."
    if len(parts) == 3 and parts[1] == "-":
        return f"The difference of {parts[0]} and {parts[2]} is {number}."
    if len(parts) == 3 and parts[1] == "*":
        return f"The product of {parts[0]} and {parts[2]} is {number}."
    return f"{text} = {number}"


def _ddgs_tool() -> Any:
    """Build a DuckDuckGo search tool backed directly by the ``ddgs`` package."""
    from ddgs import DDGS  # imported lazily: it is an optional dependency

    @tool("duckduckgo_search")
    def duckduckgo_search(query: str) -> str:
        """Search the web with DuckDuckGo for current or factual information."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=SEARCH_RESULTS))
        except Exception as exc:  # network hiccups / rate limits
            return f"Web search failed: {exc}"
        if not results:
            return "No good DuckDuckGo Search Result was found"
        lines = []
        for index, item in enumerate(results, start=1):
            title = item.get("title", "").strip()
            body = item.get("body", "").strip()
            link = item.get("href") or item.get("url") or ""
            # Trim the snippets: every extra token is another millisecond in
            # the model call that turns the results into an answer.
            lines.append(
                f"{index}. {title}\n{body[:SEARCH_SNIPPET_CHARS]}\n{link}".strip()
            )
        return "\n\n".join(lines)

    return duckduckgo_search


def build_search_tool() -> Any | None:
    """Return a web-search tool, or ``None`` when no search backend is available."""
    try:
        return _ddgs_tool()
    except ImportError:
        pass
    try:  # fallback for installs that still ship the community wrapper
        from langchain_community.tools import DuckDuckGoSearchRun

        return DuckDuckGoSearchRun()
    except Exception:
        return None


def build_tools() -> list[Any]:
    """Return the tool set exposed to the agent."""
    tools: list[Any] = [calculator]
    search_tool = build_search_tool()
    if search_tool is not None:
        tools.append(search_tool)
    return tools


# --------------------------------------------------------------------------- #
# Model + agent
# --------------------------------------------------------------------------- #
def build_model(
    api_key: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
):
    """Create the Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {
        "model": model_name or get_model_name(),
        "temperature": 0,
        "google_api_key": api_key or get_api_key(),
        # A quota error is not worth retrying for half a minute: every fallback
        # model has its own quota, so failing fast lets the caller move on.
        "max_retries": 0,
        # The API has been seen to send the response headers and then stall
        # mid-body, which would otherwise hang until the request timeout.
        "client_args": {"timeout": REQUEST_TIMEOUT},
    }
    # Thinking is what makes the flash models slow. A ReAct turn only needs
    # the tool decision and the final sentence, not a long inner monologue.
    effort = (reasoning_effort or os.getenv("REASONING_EFFORT") or "").strip()
    if effort:
        kwargs["reasoning_effort"] = effort

    model = ChatGoogleGenerativeAI(**kwargs)

    # The Google SDK retries failed requests five times with a growing
    # backoff, which turns a quota error into a 36 second wait. The fallback
    # chain is a better answer, so retries are switched off here as well.
    try:
        from google import genai
        from google.genai.types import HttpOptions, HttpRetryOptions

        model.client = genai.Client(
            api_key=kwargs["google_api_key"],
            http_options=HttpOptions(retry_options=HttpRetryOptions(attempts=1)),
        )
    except Exception:  # pragma: no cover - the SDK layout changed
        pass

    return model


def build_agent(
    model: Any | None = None,
    tools: Iterable[Any] | None = None,
    model_name: str | None = None,
):
    """Create the ReAct agent.

    Uses LangChain v1's ``create_agent`` when available and falls back to
    LangGraph's (now deprecated) ``create_react_agent`` prebuilt so the project
    still runs on older installs.
    """
    tools = list(tools) if tools is not None else build_tools()
    model = model if model is not None else build_model(model_name=model_name)

    try:
        from langchain.agents import create_agent

        return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)
    except ImportError:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)


# --------------------------------------------------------------------------- #
# Streaming helpers
# --------------------------------------------------------------------------- #
def message_text(message: Any) -> str:
    """Return the plain-text part of a message (Gemini may return content parts)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "") or ""))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return ""


def message_text_raw(message: Any) -> str:
    """The text of a message *without* stripping it.

    Streaming needs the spaces between words, so the chunks have to be kept
    exactly as the provider sent them.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "") or ""))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return ""


def tool_call_names(message: Any) -> list[str]:
    """Return the names of the tools an AI message asked for."""
    names: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name:
            names.append(str(name))
    return names


def build_messages(
    message: str,
    history: Iterable[tuple[str, str]] | None = None,
) -> list:
    """Build the message list handed to the agent.

    ``history`` is a list of ``(role, content)`` pairs where role is ``"user"``
    or ``"assistant"``. They are replayed before the new message so the agent
    can answer follow-up questions ("and what tools does it have?") instead of
    starting from scratch on every turn.
    """
    messages: list = []
    for role, content in history or []:
        content = (content or "").strip()
        if not content:
            continue
        if role == "assistant":
            # A history that starts with an assistant turn (it can happen after
            # the window was trimmed) would leave the model without a question.
            if not messages:
                continue
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=message))
    return messages


def stream_reply_events(
    agent: Any,
    message: str,
    history: Iterable[tuple[str, str]] | None = None,
):
    """Yield ``("delta", text)`` per visible token, then ``("done", reply, tools)``.

    The agent is streamed in "messages" mode so the answer reaches the browser
    while it is being written, instead of after the whole turn has finished.
    """
    tools_used: list[str] = []
    narrations: list[str] = []
    final_text = ""

    for chunk in agent.stream(
        {"messages": build_messages(message, history)}, stream_mode="messages"
    ):
        message_chunk = chunk[0] if isinstance(chunk, tuple) else chunk

        if isinstance(message_chunk, ToolMessage):
            name = getattr(message_chunk, "name", None)
            if name:
                tools_used.append(str(name))
            continue
        if not isinstance(message_chunk, AIMessage):
            continue

        calls = tool_call_names(message_chunk)
        text = message_text_raw(message_chunk)
        if calls:
            tools_used.extend(calls)
            narrations.append(message_text(message_chunk))
            continue
        if not text:
            continue

        # Providers differ: some chunks carry the text so far, others only the
        # new part. Handle both so the client always receives clean deltas.
        if final_text and text.startswith(final_text):
            delta = text[len(final_text):]
            final_text = text
        elif final_text and final_text.startswith(text):
            # the model rewrote the text
            delta = ""
            final_text = text
        else:
            delta = text
            final_text += text

        if delta:
            yield ("delta", delta)

    reply = (final_text or ("\n\n".join(narrations) if narrations else "")).strip()
    if not reply:
        reply = "(the agent returned an empty response)"
    yield ("done", reply, sorted(set(tools_used)))


def stream_reply(
    agent: Any,
    message: str,
    history: Iterable[tuple[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """Run the agent on ``message`` and return ``(reply, tools_used)``.

    Works with both agent flavours: the streamed updates use the node name
    ``"model"`` (LangChain v1) or ``"agent"`` (LangGraph prebuilt), so the
    chunks are inspected generically.

    ``history`` replays earlier turns so the agent keeps the conversation
    context. Only the *new* messages appear in the streamed updates, so the
    history never leaks into the returned reply.
    """
    tools_used: list[str] = []
    narrations: list[str] = []
    final_text = ""

    for chunk in agent.stream(
        {"messages": build_messages(message, history)}, stream_mode="updates"
    ):
        if not isinstance(chunk, dict):
            continue
        for payload in chunk.values():
            if not isinstance(payload, dict):
                continue
            for msg in payload.get("messages") or []:
                if isinstance(msg, ToolMessage):
                    name = getattr(msg, "name", None)
                    if name:
                        tools_used.append(str(name))
                    continue
                if not isinstance(msg, AIMessage):
                    continue
                text = message_text(msg)
                calls = tool_call_names(msg)
                if calls:
                    tools_used.extend(calls)
                    if text:
                        narrations.append(text)
                    continue
                if text:
                    final_text = text

    reply = final_text or ("\n\n".join(narrations) if narrations else "")
    if not reply:
        reply = "(the agent returned an empty response)"
    return reply, sorted(set(tools_used))
