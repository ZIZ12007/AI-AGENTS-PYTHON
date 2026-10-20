"""Terminal ReAct agent (Gemini + DuckDuckGo search + calculator).

Run:
    pip install -r requirements.txt
    # .env -> GOOGLE_API_KEY=...
    python MAIN.PY
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent_core import (  # noqa: E402  (path setup has to run first)
    MAX_HISTORY_TURNS,
    build_agent,
    get_api_key,
    get_model_name,
    stream_reply,
)


def main() -> None:
    if not get_api_key():
        print("Error: GOOGLE_API_KEY not found in .env file.")
        print("Create Project1/.env with:  GOOGLE_API_KEY=your_key_here")
        return

    print(f"Loading agent (model: {get_model_name()})...")
    try:
        agent = build_agent()
    except Exception as exc:
        print(f"Error: could not start the agent -> {exc}")
        return

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")

    # Previous turns are replayed to the agent so follow-up questions keep the
    # context of the conversation.
    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            reply, tools_used = stream_reply(agent, user_input, history)
            print(reply)
            if tools_used:
                print(f"[tools used: {', '.join(tools_used)}]")

            history.append(("user", user_input))
            history.append(("assistant", reply))
            if len(history) > MAX_HISTORY_TURNS:
                history = history[-MAX_HISTORY_TURNS:]
        except Exception as exc:
            print(f"\nAn error occurred: {exc}")


if __name__ == "__main__":
    main()
