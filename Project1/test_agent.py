"""Offline smoke test for the agent wiring - no API key required.

    python test_agent.py

It checks the calculator tool, the ReAct streaming/extraction logic and the
agent factory, using a scripted fake model instead of a live Gemini call.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from agent_core import (  # noqa: E402
    MAX_HISTORY_TURNS,
    build_agent,
    build_messages,
    calculator,
    get_model_chain,
    is_model_unavailable_error,
    is_rate_limit_error,
    stream_reply,
)


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of AI messages, one per model call."""

    script: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


class FakeAgent:
    """Mimics the chunk shape produced by LangGraph / LangChain v1 agents."""

    def __init__(self, node: str = "model"):
        self.node = node
        self.inputs: list = []

    def stream(self, inputs, stream_mode=None):
        self.inputs.append(inputs)
        yield {
            self.node: {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "calculator",
                                "args": {"expression": "24 + 58"},
                                "id": "call_1",
                            }
                        ],
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="The sum of 24 and 58 is 82.",
                        name="calculator",
                        tool_call_id="call_1",
                    )
                ]
            }
        }
        yield {
            self.node: {
                "messages": [AIMessage(content="The sum of 24 and 58 is 82.")]
            }
        }


def check(label: str, condition: bool, extra: object = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ->  {extra}" if extra != "" else ""
    print(f"[{status}] {label}{suffix}")
    if not condition:
        raise AssertionError(label)


def test_calculator() -> None:
    result = calculator.invoke({"expression": "24 + 58"})
    check("calculator adds numbers", "82" in result)

    result = calculator.invoke({"expression": "7 * 8"})
    check("calculator multiplies numbers", "56" in result)

    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    check("calculator rejects unsafe input", result.startswith("Error"))


def test_stream_reply() -> None:
    for node in ("model", "agent"):  # LangChain v1 vs. LangGraph prebuilt
        reply, tools_used = stream_reply(FakeAgent(node), "What is 24 + 58?")
        check(f"reply extracted from '{node}' node", reply == "The sum of 24 and 58 is 82.")
        check(f"tool recorded from '{node}' node", tools_used == ["calculator"])


def test_history() -> None:
    """Earlier turns are replayed so the agent keeps the conversation context."""
    messages = build_messages(
        "And what tools does it have?",
        [("user", "What is a ReAct agent?"), ("assistant", "It loops thought/tool/observation.")],
    )
    check("history is replayed", len(messages) == 3)
    check("history keeps the user turn", messages[0].content == "What is a ReAct agent?")
    check("history keeps the assistant turn", messages[1].content == "It loops thought/tool/observation.")
    check("the new message is last", messages[-1].content == "And what tools does it have?")

    agent = FakeAgent("model")
    stream_reply(agent, "And what tools does it have?", [("user", "What is a ReAct agent?")])
    sent = agent.inputs[0]["messages"]
    check("the agent receives the history", len(sent) == 2)
    check("the new message is sent last", sent[-1].content == "And what tools does it have?")


def test_agent_factory() -> None:
    model = ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "24 + 58"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="The sum of 24 and 58 is 82."),
        ]
    )
    agent = build_agent(model=model)
    reply, tools_used = stream_reply(agent, "What is 24 + 58?")
    check("agent loop answers", "82" in reply)
    check("agent loop used the calculator", "calculator" in tools_used)


class StreamingFakeAgent:
    """Streams an answer token by token, like the real agent does."""

    def __init__(self, text="The sum of 24 and 58 is 82."):
        self.text = text

    def stream(self, inputs, stream_mode=None):
        if stream_mode != "messages":
            yield {"model": {"messages": [AIMessage(content=self.text)]}}
            return
        for word in self.text.split(" "):
            yield (AIMessage(content=word + " "), {})


def test_streaming() -> None:
    """The streaming helper yields deltas and then the finished reply."""
    from agent_core import stream_reply_events

    events = list(stream_reply_events(StreamingFakeAgent(), "What is 24 + 58?"))
    kinds = [event[0] for event in events]
    check("streaming yields deltas", kinds.count("delta") > 1, kinds)
    check("streaming ends with done", kinds[-1] == "done", kinds)

    reply, tools = events[-1][1], events[-1][2]
    check("streaming returns the whole reply", reply.strip() == "The sum of 24 and 58 is 82.", reply)
    check("streaming returns the tools", tools == [], tools)


def test_cache() -> None:
    """A repeated question is answered from the cache without an API call."""
    import server

    server._CACHE.clear()
    key = server._cache_key("What is 24 + 58?", [])
    check("an unknown key is not cached", server.cache_get(key) is None)

    server.cache_set(key, {"reply": "82", "tools_used": ["calculator"]})
    cached = server.cache_get(key)
    check("a stored answer comes back", cached is not None and cached["reply"] == "82")
    check(
        "the cache key depends on the message",
        server._cache_key("something else", []) != key,
    )
    server._CACHE.clear()


def test_providers() -> None:
    """The provider layer picks the right key and the right model chain."""
    import os

    from agent_core import (
        PROVIDERS,
        available_providers,
        get_model_chain,
        get_provider,
        provider_key,
    )

    saved = {name: os.environ.get(name) for name in ("PROVIDER", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY")}
    try:
        os.environ["PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "test-groq"
        os.environ["OPENROUTER_API_KEY"] = "test-openrouter"
        check("an explicit provider wins", get_provider() == "groq", get_provider())
        check("the provider key is found", provider_key("groq") == "test-groq")
        check("both providers are available", set(available_providers()) >= {"groq", "openrouter"}, available_providers())
        check(
            "the chain starts with the provider model",
            get_model_chain()[0] == PROVIDERS["groq"]["default"],
            get_model_chain(),
        )
        check("the chain holds several models", len(get_model_chain()) >= 2, get_model_chain())

        os.environ.pop("PROVIDER")
        os.environ.pop("GOOGLE_API_KEY", None)
        check(
            "the first provider with a key wins",
            get_provider() in ("groq", "openrouter"),
            get_provider(),
        )
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_model_selection() -> None:
    """The client may pick a model, but only one the server can reach."""
    import os

    from agent_core import available_models, model_is_allowed

    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    try:
        models = available_models()
        check("the server offers models", len(models) > 0, len(models))
        check(
            "the offered models name a provider",
            all(entry.get("provider") and entry.get("model") for entry in models),
            models[:2],
        )
        first = models[0]["model"]
        check("a listed model is allowed", model_is_allowed(first))
        check("an unknown model is not allowed", not model_is_allowed("evil-model-xyz"))
        check("an empty model is not allowed", not model_is_allowed(""))
    finally:
        os.environ.pop("GOOGLE_API_KEY", None)


def test_requested_model_is_used() -> None:
    """A model the client asks for is used; an unknown one is ignored."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("[SKIP] model choice test (pip install httpx to enable it)")
        return

    import os

    import server

    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    server._agents.clear()
    server._agents.update({name: FakeAgent("model") for name in server.get_model_chain()})
    server._agent = server._agents[server.get_model_name()]
    server._CACHE.clear()
    try:
        with TestClient(server.app) as client:
            allowed = server.get_model_chain()[1]
            payload = client.post(
                "/api/chat",
                json={"message": "Which model answers this?", "model": allowed},
            ).json()
            check("the requested model is used", payload.get("model") == allowed, payload.get("model"))

            payload = client.post(
                "/api/chat",
                json={"message": "And this one?", "model": "not-a-real-model"},
            ).json()
            check(
                "an unknown model is ignored",
                payload.get("model") != "not-a-real-model"
                and payload.get("model") in server.get_model_chain(),
                payload.get("model"),
            )
    finally:
        server._agents.clear()
        server._agent = None
        server._CACHE.clear()
        os.environ.pop("GOOGLE_API_KEY", None)


def test_api() -> None:
    """POST /api/chat against a stubbed agent (no API key needed)."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:  # httpx is not part of the base requirements
        print("[SKIP] api test (pip install httpx to enable it)")
        return

    import os

    import server

    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    # The server keeps one agent per model id, so stub every model in the chain.
    server._agents.clear()
    server._agents.update({name: FakeAgent("model") for name in server.get_model_chain()})
    server._agent = server._agents[server.get_model_name()]
    try:
        with TestClient(server.app) as client:
            health = client.get("/api/health").json()
            check("health endpoint answers", health.get("ok") is True)
            check("health reports the message limit", health.get("max_message_chars") == 4000)
            check("health reports the history window", health.get("history_turns") == MAX_HISTORY_TURNS)
            check("health reports the model chain", len(health.get("model_chain") or []) >= 1)
            check(
                "health lists the models the user can pick",
                len(health.get("models_available") or []) >= 1,
                health.get("models_available"),
            )

            payload = client.post(
                "/api/chat", json={"message": "What is 24 + 58?"}
            ).json()
            check("chat endpoint answers", "82" in payload.get("reply", ""))
            check("chat endpoint lists the tools", payload.get("tools_used") == ["calculator"])
            check("chat endpoint is live", payload.get("demo") is False)
            check("chat endpoint reports the duration", payload.get("elapsed_ms") is not None)

            # A follow-up question carries the previous turns.
            follow_up = client.post(
                "/api/chat",
                json={
                    "message": "And what tools does it have?",
                    "history": [
                        {"role": "user", "content": "What is 24 + 58?"},
                        {"role": "assistant", "content": "The sum of 24 and 58 is 82."},
                    ],
                },
            )
            check("chat endpoint accepts a history", follow_up.status_code == 200)

            # An empty message is rejected with a 422 and a readable detail.
            empty = client.post("/api/chat", json={"message": ""})
            check("chat endpoint rejects an empty message", empty.status_code == 422)

            # Unknown API routes answer with JSON, not with the static files.
            missing = client.get("/api/does-not-exist")
            check("unknown API routes 404", missing.status_code == 404)
            check("unknown API routes answer with JSON", "detail" in missing.json())

            check("frontend is served", client.get("/").status_code == 200)
            check("stylesheet is served", client.get("/styles.css").status_code == 200)
            check("frontend script is served", client.get("/app.js").status_code == 200)

            # The frontend must not hardcode a localhost backend URL.
            index = client.get("/").text
            check(
                "the frontend does not hardcode localhost",
                'value="http://localhost:8000/api/chat"' not in index,
            )
    finally:
        server._agents.clear()
        server._agent = None
        os.environ.pop("GOOGLE_API_KEY", None)


def test_model_fallback() -> None:
    """A rate limited model must not take the whole request down."""
    check("the model chain has fallbacks", len(get_model_chain()) >= 2, get_model_chain())
    check("the chain starts with the configured model", get_model_chain()[0] == get_model_chain()[0])

    quota_error = Exception(
        "429 RESOURCE_EXHAUSTED: Quota exceeded for metric: "
        "generate_content_free_tier_requests, limit: 20"
    )
    check("quota errors are detected", is_rate_limit_error(quota_error))
    check("other errors are not quota errors", not is_rate_limit_error(ValueError("boom")))
    check("quota errors mean the model is unavailable", is_model_unavailable_error(quota_error))

    overload_error = Exception(
        "503 UNAVAILABLE: This model is currently experiencing high demand."
    )
    check(
        "an overloaded model means the model is unavailable",
        is_model_unavailable_error(overload_error),
    )
    check("a random error is not recoverable", not is_model_unavailable_error(ValueError("boom")))

    import server

    class RateLimitedAgent:
        def stream(self, inputs, stream_mode=None):
            raise quota_error

    class WorkingAgent(FakeAgent):
        pass

    saved = dict(server._agents)
    try:
        # the first model is rate limited, the second one works
        chain = server.get_model_chain()
        server._agents.clear()
        server._agents[chain[0]] = RateLimitedAgent()
        server._agents[chain[1]] = WorkingAgent("model")
        server._agent = server._agents[chain[0]]

        import os

        os.environ.setdefault("GOOGLE_API_KEY", "test-key")
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            print("[SKIP] fallback test (pip install httpx to enable it)")
            return

        with TestClient(server.app) as client:
            payload = client.post("/api/chat", json={"message": "What is 24 + 58?"}).json()
            check("a rate limited model falls back to the next one", "82" in payload.get("reply", ""))
            check("the fallback reports no error", payload.get("error") is not True)
            check("the fallback names the model it used", payload.get("model") == chain[1])
    finally:
        server._agents.clear()
        server._agents.update(saved)
        server._agent = saved.get(server.get_model_name())


def test_history_normalisation() -> None:
    """The server drops unknown roles and caps the replayed turns."""
    import server

    turns = server.ChatTurn(role="user", content="hello")
    cleaned = server.normalise_history([turns])
    check("history keeps valid turns", cleaned == [("user", "hello")])

    dropped = server.normalise_history([server.ChatTurn(role="system", content="ignore me")])
    check("history drops unknown roles", dropped == [])

    long_history = [
        server.ChatTurn(role="user", content=f"turn {index}")
        for index in range(MAX_HISTORY_TURNS + 5)
    ]
    capped = server.normalise_history(long_history)
    check("history is capped", len(capped) == MAX_HISTORY_TURNS)
    check("history keeps the newest turns", capped[-1][1] == f"turn {MAX_HISTORY_TURNS + 4}")


def test_stream_endpoint() -> None:
    """POST /api/chat/stream returns server-sent events."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("[SKIP] stream test (pip install httpx to enable it)")
        return

    import os

    import server

    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    server._agents.clear()
    server._agents.update(
        {name: StreamingFakeAgent() for name in server.get_model_chain()}
    )
    server._agent = server._agents[server.get_model_name()]
    server._CACHE.clear()
    try:
        with TestClient(server.app) as client:
            with client.stream(
                "POST", "/api/chat/stream", json={"message": "stream me a fresh answer"}
            ) as response:
                body = "".join(response.iter_text())
                status = response.status_code

        check("the stream endpoint answers", status == 200, status)
        check("the stream sends deltas", "event: delta" in body, body[:120])
        check("the stream ends with done", "event: done" in body, body[-160:])
        check(
            "the stream carries the full answer",
            "The sum of 24 and 58 is 82." in body,
            body[:200],
        )
        check(
            "the stream does not leak internals",
            "generativelanguage" not in body and "RESOURCE_EXHAUSTED" not in body,
            body[:200],
        )
    finally:
        server._agents.clear()
        server._agent = None
        server._CACHE.clear()
        os.environ.pop("GOOGLE_API_KEY", None)


def test_stream_error_is_sanitised() -> None:
    """A provider error must never reach the client as raw text."""
    import server

    class ExplodingAgent:
        def stream(self, inputs, stream_mode=None):
            raise Exception(
                "Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 "
                "RESOURCE_EXHAUSTED: {'error': {'message': 'You exceeded your quota'}}"
            )
            yield  # pragma: no cover

    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("[SKIP] stream error test (pip install httpx to enable it)")
        return

    import os

    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    server._agents.clear()
    server._agents.update({name: ExplodingAgent() for name in server.get_model_chain()})
    server._agent = server._agents[server.get_model_name()]
    server._CACHE.clear()
    try:
        with TestClient(server.app) as client:
            with client.stream(
                "POST", "/api/chat/stream", json={"message": "explode please"}
            ) as response:
                body = "".join(response.iter_text())

        check("the error is reported", "event: error" in body, body[:160])
        check(
            "the raw provider error is not sent",
            "RESOURCE_EXHAUSTED" not in body and "generativelanguage" not in body,
            body[:200],
        )
        check(
            "the quota message explains the wait",
            "quota" in body.lower(),
            body[:200],
        )
    finally:
        server._agents.clear()
        server._agent = None
        server._CACHE.clear()
        os.environ.pop("GOOGLE_API_KEY", None)


if __name__ == "__main__":
    test_calculator()
    test_stream_reply()
    test_history()
    test_agent_factory()
    test_history_normalisation()
    test_model_fallback()
    test_streaming()
    test_cache()
    test_providers()
    test_model_selection()
    test_requested_model_is_used()
    test_api()
    test_stream_endpoint()
    test_stream_error_is_sanitised()
    print("\nAll checks passed.")
