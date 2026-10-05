# Portfolio write-up (copy-paste)

## Title
ReAct AI Agent — Gemini + Web Search + Minimal Chat UI

## One-liner
A Python ReAct agent that reasons with tools (live web search + calculator),
wrapped in a minimal chat frontend with a FastAPI backend and offline demo mode.

## Description (for portfolio / LinkedIn / resume)
Built a ReAct-style AI agent in Python using LangChain, LangGraph, and Gemini
(`gemini-2.0-flash`). The agent loops Thought → Action → Observation with two
tools: DuckDuckGo web search and a calculator. Added a minimal single-column
chat UI (vanilla JS, no framework) that calls a FastAPI backend
(`POST /api/chat`), displays which tools fired per answer, and gracefully
falls back to a local demo when the backend is offline. Shipped with one-click
deploys: GitHub Pages (frontend demo), Render (full backend), Vercel.

## Highlights (bullets)
- ReAct loop with tool-calling (search + calculator) on Gemini
- FastAPI wrapper with `/api/chat` + `/api/health`, CORS, static serving
- Minimal chat UX: tool badges, example prompts, settings, demo fallback
- Live demo via GitHub Pages; full-stack via Render blueprint
- Clean docs: README, screenshots, architecture diagram, demo script

## Tech tags
Python, LangChain, LangGraph, Google Gemini, DuckDuckGo, FastAPI, Uvicorn,
JavaScript, HTML, CSS, GitHub Actions, Render, Vercel

## Links (fill after deploy)
- Live demo: https://ZIZ12007.github.io/AI-AGENTS-PYTHON/
- Full-stack API: https://<your-service>.onrender.com/api/health
- Code: https://github.com/ZIZ12007/AI-AGENTS-PYTHON
- Screenshots: docs/screenshots/desktop.png, docs/screenshots/mobile.png

## Resume line
Built and deployed a ReAct AI agent (Python, LangGraph, Gemini) with web-search
and calculator tools plus a minimal chat frontend and FastAPI backend.
