"""Minimal FastAPI wrapper around MAIN.PY ReAct agent.

Run:
    pip install -r requirements.txt
    # create .env with GOOGLE_API_KEY=...
    uvicorn server:app --reload --port 8000

Frontend in ./frontend/ talks to POST /api/chat.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="AI Agent — Minimal Frontend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic operations."""
    return f"The sum of {a} and {b} is {a + b}."


_agent = None


def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=api_key,
    )
    tools = [DuckDuckGoSearchRun(), calculator]
    _agent = create_react_agent(model=model, tools=tools)
    return _agent


class ChatIn(BaseModel):
    message: str


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "has_key": bool(os.getenv("GOOGLE_API_KEY")),
        "model": "gemini-2.0-flash",
        "tools": ["duckduckgo_search", "calculator"],
    }


@app.post("/api/chat")
def chat(body: ChatIn):
    agent = get_agent()
    if agent is None:
        return {
            "reply": "GOOGLE_API_KEY is missing on the server. Add it to Project1/.env then restart.",
            "tools_used": [],
            "demo": True,
        }
    tools_used: list[str] = []
    chunks: list[str] = []
    for chunk in agent.stream({"messages": [HumanMessage(content=body.message)]}):
        # tool calls
        if "tools" in chunk and "messages" in chunk["tools"]:
            for m in chunk["tools"]["messages"]:
                tools_used.append(getattr(m, "name", "tool"))
        if "agent" in chunk and "messages" in chunk["agent"]:
            for message in chunk["agent"]["messages"]:
                content = message.content
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                if content:
                    # skip pure tool-call placeholder messages
                    if getattr(message, "tool_calls", None):
                        tools_used.append(
                            message.tool_calls[0].get("name", "tool")
                            if isinstance(message.tool_calls[0], dict)
                            else "tool"
                        )
                        continue
                    chunks.append(str(content))
    reply = "".join(chunks).strip() or "(empty response)"
    return {"reply": reply, "tools_used": sorted(set(tools_used)), "demo": False}


# Serve minimal frontend at /  (if folder exists)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
