/* Minimal chat: POSTs to FastAPI backend, falls back to local demo. */
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const dot = document.getElementById("dot");
const statusText = document.getElementById("statusText");
const modeNote = document.getElementById("modeNote");
const clearBtn = document.getElementById("clearBtn");
const settingsBtn = document.getElementById("settingsBtn");
const settings = document.getElementById("settings");
const backendUrl = document.getElementById("backendUrl");

let backend = localStorage.getItem("agent-backend") || backendUrl.value;
backendUrl.value = backend;
let demoOnly = localStorage.getItem("agent-demo-only") === "1";
let live = false;

function setStatus(mode) {
  live = mode === "live";
  dot.className = "dot " + (live ? "live" : "demo");
  statusText.textContent = live ? "connected" : "demo mode";
  modeNote.textContent = live ? "connected to backend" : "demo fallback on";
}

function bubble(who, text, tools = []) {
  const d = document.createElement("div");
  d.className = "msg " + who;
  d.textContent = text;
  if (tools.length) {
    const m = document.createElement("span");
    m.className = "meta";
    m.innerHTML = tools.map(t => `<span class="badge">⚙ ${t}</span>`).join("");
    d.appendChild(m);
  }
  chat.appendChild(d);
  d.scrollIntoView({ behavior: "smooth", block: "end" });
  return d;
}

function typing() {
  const d = document.createElement("div");
  d.className = "msg agent typing";
  d.textContent = "···";
  chat.appendChild(d);
  return d;
}

// --- local demo (mirrors MAIN.PY tools, no key needed) ---
function demoReply(q) {
  const m = q.match(/(-?\d+(?:\.\d+)?)\s*\+\s*(-?\d+(?:\.\d+)?)/);
  if (m) {
    const a = parseFloat(m[1]), b = parseFloat(m[2]);
    return { reply: `The sum of ${a} and ${b} is ${a + b}.`, tools: ["calculator"] };
  }
  if (/react agent/i.test(q)) {
    return { reply: "A ReAct agent loops: Thought → Action (tool) → Observation, until it can answer. This one has web search + calculator tools on a Gemini model.", tools: [] };
  }
  if (/gemini|news|search/i.test(q)) {
    return { reply: "Demo mode: connect the backend (uvicorn server:app) to get live DuckDuckGo search + Gemini answers. Your query would run through the ReAct loop.", tools: ["duckduckgo_search"] };
  }
  if (/^(hi|hello|hey)\b/i.test(q.trim())) {
    return { reply: "Hello! I'm your AI assistant (demo mode). Ask me a sum like “What is 24 + 58?” or start the backend for full search + Gemini.", tools: [] };
  }
  return { reply: `Demo reply: “${q}”\n\nStart the backend for real answers:\n  pip install -r requirements.txt\n  uvicorn server:app --port 8000`, tools: [] };
}

async function send(text) {
  const q = (text ?? input.value).trim();
  if (!q) return;
  bubble("user", q);
  input.value = "";
  sendBtn.disabled = true;
  const t = typing();

  if (!demoOnly) {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 60000);
      const res = await fetch(backend, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q }),
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      t.remove();
      bubble("agent", data.reply || "(empty)", data.tools_used || []);
      setStatus(data.demo ? "demo" : "live");
      try { localStorage.setItem("agent-backend", backend); } catch {}
      sendBtn.disabled = false;
      return;
    } catch (e) {
      // fall through to demo
    }
  }
  await new Promise(r => setTimeout(r, 450));
  t.remove();
  const d = demoReply(q);
  bubble("agent", d.reply, d.tools);
  setStatus("demo");
  sendBtn.disabled = false;
}

form.addEventListener("submit", e => { e.preventDefault(); send(); });
document.querySelectorAll("[data-ex]").forEach(b =>
  b.addEventListener("click", () => send(b.dataset.ex)));
clearBtn.addEventListener("click", () => { chat.innerHTML = ""; seed(); });
settingsBtn.addEventListener("click", () => { settings.hidden = !settings.hidden; });
document.getElementById("saveSettings").addEventListener("click", () => {
  backend = backendUrl.value.trim() || backend;
  demoOnly = false;
  try {
    localStorage.setItem("agent-backend", backend);
    localStorage.removeItem("agent-demo-only");
  } catch {}
  settings.hidden = true;
  bubble("agent", `Backend set to ${backend}. Next message will try the live agent.`);
});
document.getElementById("demoBtn").addEventListener("click", () => {
  demoOnly = true;
  try { localStorage.setItem("agent-demo-only", "1"); } catch {}
  settings.hidden = true;
  setStatus("demo");
});

function seed() {
  bubble("agent", "Welcome! I'm your AI assistant. Ask a math question or anything to search.\nType 'quit' in the terminal version — here just keep chatting.");
}
setStatus("demo");
seed();
