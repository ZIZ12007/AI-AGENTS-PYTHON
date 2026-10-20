/* Minimal chat: talks to the FastAPI backend, falls back to a local demo. */
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
const healthInfo = document.getElementById("healthInfo");
const resetBtn = document.getElementById("resetBtn");
const counter = document.getElementById("counter");
const modelSelect = document.getElementById("modelSelect");

// Keep in sync with server.py (MAX_MESSAGE_CHARS / MAX_HISTORY_TURNS).
const MAX_MESSAGE_CHARS = 4000;
const MAX_HISTORY_TURNS = 12;
// The server reports its own budget in /api/health; the client waits a little
// longer than that. This fallback is used until the health probe answers.
let serverTimeoutSeconds = 120;
const CHAT_TIMEOUT_FLOOR_MS = 90000;
const CHAT_TIMEOUT_PADDING_S = 15;
const HEALTH_TIMEOUT_MS = 8000;

const STORAGE_BACKEND = "agent-backend";
const STORAGE_DEMO_ONLY = "agent-demo-only";
const STORAGE_MODEL = "agent-model";

const STATUS = {
  connecting: { cls: "",       text: "connecting…",       note: "checking the backend" },
  live:       { cls: "live",   text: "connected",         note: "connected to backend" },
  nokey:      { cls: "demo",   text: "no api key",        note: "server has no GOOGLE_API_KEY" },
  offline:    { cls: "offline", text: "backend offline",  note: "backend unreachable" },
  demo:       { cls: "demo",   text: "demo mode",         note: "demo fallback on" },
  static:     { cls: "demo",   text: "demo mode",         note: "no backend configured - local demo" },
  error:      { cls: "error",  text: "error",             note: "the agent reported an error" },
};

let backend = normaliseChatUrl(readStored(STORAGE_BACKEND) || defaultBackend());
let demoOnly = readStored(STORAGE_DEMO_ONLY) === "1";
let history = [];
// True once a backend was chosen explicitly (settings panel). A static host
// with no backend is an expected state, not an outage.
let backendChosen = Boolean(readStored(STORAGE_BACKEND));
// The model the user picked in the settings ("" = let the server decide).
let selectedModel = readStored(STORAGE_MODEL) || "";

// --- storage (private mode throws, so every access is guarded) -------------
function readStored(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeStored(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode */ }
}
function removeStored(key) {
  try { localStorage.removeItem(key); } catch { /* private mode */ }
}

// --- backend url handling --------------------------------------------------
// A static host (GitHub Pages, Vercel) can point the demo at a hosted
// backend through config.js; everything else uses the API on its own origin.
function configuredBackend() {
  const config = window.AGENT_CONFIG || {};
  return (config.backendUrl || "").trim();
}

// Is a backend actually expected at this point? A static-only deployment has
// none, and that is a normal state rather than an outage.
function hasBackend() {
  return backendChosen || Boolean(configuredBackend());
}

// The FastAPI backend serves this very page, so the default is the API on the
// same origin. Only a page opened straight from disk needs an absolute URL.
function defaultBackend() {
  const configured = configuredBackend();
  if (configured) return normaliseChatUrl(configured);
  if (location.protocol === "file:") return "http://localhost:8000/api/chat";
  return "/api/chat";
}

// Accepts a full chat URL, a base URL ("https://host" or "https://host:8000")
// or an "/api" URL and always ends up pointing at the chat endpoint.
function normaliseChatUrl(raw) {
  const value = (raw || "").trim();
  if (!value) return defaultBackend();
  const trimmed = value.replace(/\/+$/, "");
  if (/\/api\/chat$/i.test(trimmed)) return trimmed;
  if (/\/api$/i.test(trimmed)) return trimmed + "/chat";
  return trimmed + "/api/chat";
}

function healthUrl() {
  return normaliseChatUrl(backend).replace(/\/api\/chat$/i, "/api/health");
}

// The streaming route sits next to the plain chat route.
function chatStreamUrl() {
  return normaliseChatUrl(backend).replace(/\/api\/chat$/i, "/api/chat/stream");
}

// --- model picker ----------------------------------------------------------
// The server reports every model its keys can reach, grouped by provider.
function fillModelPicker(models, activeModel) {
  if (!modelSelect) return;
  modelSelect.textContent = "";

  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = "Automatic (fastest available)";
  modelSelect.appendChild(auto);

  const groups = new Map();
  (models || []).forEach((entry) => {
    const key = entry.provider_label || entry.provider;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(entry);
  });

  groups.forEach((entries, label) => {
    const group = document.createElement("optgroup");
    group.label = label;
    entries.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.model;
      option.textContent = entry.model + (entry.default ? "  (default)" : "");
      group.appendChild(option);
    });
    modelSelect.appendChild(group);
  });

  // Keep the stored choice if it is still offered.
  const values = Array.from(modelSelect.options).map((option) => option.value);
  if (selectedModel && values.includes(selectedModel)) {
    modelSelect.value = selectedModel;
  } else {
    selectedModel = "";
    modelSelect.value = "";
  }
  if (activeModel) {
    modelSelect.title = `The server is currently using ${activeModel}.`;
  }
}

// --- status + health -------------------------------------------------------
function setStatus(state, data) {
  const s = STATUS[state] || STATUS.demo;
  dot.className = "dot " + s.cls;
  statusText.textContent = s.text;
  modeNote.textContent = s.note;
  renderHealth(state, data);
}

function renderHealth(state, data) {
  if (!healthInfo) return;
  healthInfo.textContent = "";
  if (demoOnly) {
    const note = document.createElement("p");
    note.className = "health-note";
    note.textContent = "Demo only mode is on - the backend is not called.";
    healthInfo.appendChild(note);
    return;
  }
  if (!data) {
    const note = document.createElement("p");
    note.className = "health-note";
    note.textContent = state === "offline"
      ? `No response from ${backend}.`
      : state === "static"
        ? `No backend is configured, so the local demo is used. Set one in config.js or the ⚙ settings.`
        : "Health details appear once the backend answers.";
    healthInfo.appendChild(note);
    return;
  }
  const rows = [
    ["Backend", backend],
    ["Provider", data.provider_label || data.provider || "unknown"],
    ["Model", data.model || "unknown"],
    ["Tools", (data.tools || []).join(", ") || "none"],
    ["API key", data.has_key ? "configured" : "missing"],
    ["Timeout", data.timeout_seconds ? `${data.timeout_seconds}s` : "unknown"],
  ];
  const chain = data.model_chain || [];
  if (chain.length > 1) {
    rows.push(["Fallbacks", chain.slice(1).join(", ")]);
  }
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "health-row";
    const key = document.createElement("span");
    key.className = "health-key";
    key.textContent = label;
    const val = document.createElement("span");
    val.className = "health-val";
    val.textContent = value;
    row.append(key, val);
    healthInfo.appendChild(row);
  });
}

async function probeHealth() {
  if (demoOnly) {
    setStatus("demo");
    return null;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
  try {
    const res = await fetch(healthUrl(), {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    if (data.timeout_seconds) serverTimeoutSeconds = Number(data.timeout_seconds);
    fillModelPicker(data.models_available, data.model);
    setStatus(data.has_key ? "live" : "nokey", data);
    return data;
  } catch {
    // A static host with no backend configured is a normal state, not an
    // outage - say so instead of showing "backend offline".
    setStatus(hasBackend() ? "offline" : "static", null);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// --- chat rendering --------------------------------------------------------
function bubble(who, text, tools = []) {
  const d = document.createElement("div");
  d.className = "msg " + who;
  d.textContent = text;
  if (tools.length) {
    const m = document.createElement("span");
    m.className = "meta";
    tools.forEach((name) => {
      // Built as DOM nodes: the tool names come from the server.
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "⚙ " + name;
      m.appendChild(badge);
    });
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

// --- conversation history --------------------------------------------------
function recordTurn(role, content) {
  const text = (content || "").trim();
  if (!text) return;
  history.push({ role, content: text });
  if (history.length > MAX_HISTORY_TURNS) {
    history = history.slice(-MAX_HISTORY_TURNS);
  }
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

// --- backend call ----------------------------------------------------------
function describeError(status, payload) {
  if (payload) {
    if (typeof payload.detail === "string" && payload.detail) return payload.detail;
    if (Array.isArray(payload.detail)) {
      const parts = payload.detail
        .map((item) => (item && item.msg) || String(item))
        .filter(Boolean);
      if (parts.length) return parts.join("; ");
    }
    if (typeof payload.reply === "string" && payload.reply) return payload.reply;
  }
  return `The backend returned HTTP ${status}.`;
}

// Streamed chat: the answer appears while it is written. Falls back to the
// plain JSON endpoint when the server has no streaming route.
async function callBackendStreaming(q, onDelta) {
  const controller = new AbortController();
  const budgetMs = Math.max(
    CHAT_TIMEOUT_FLOOR_MS,
    (serverTimeoutSeconds + CHAT_TIMEOUT_PADDING_S) * 1000,
  );
  const timer = setTimeout(() => controller.abort(), budgetMs);
  try {
    const response = await fetch(chatStreamUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        message: q,
        history: history.slice(-MAX_HISTORY_TURNS),
        model: selectedModel || undefined,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      return { ok: false, offline: false, message: describeError(response.status, payload) };
    }
    if (!response.body) {
      return { ok: false, offline: false, message: "The backend sent no stream." };
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let reply = "";
    let tools = [];

    const handle = (event, data) => {
      if (event === "delta") {
        reply += data.text || "";
        onDelta(data.text || "");
      } else if (event === "done") {
        reply = data.reply || reply;
        tools = data.tools_used || tools;
      } else if (event === "error") {
        throw new Error(data.message || "The agent failed.");
      }
    };

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        let event = "message";
        const dataLines = [];
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (dataLines.length) {
          let parsed = null;
          try { parsed = JSON.parse(dataLines.join(" ")); } catch { parsed = null; }
          if (parsed) handle(event, parsed);
        }
        split = buffer.indexOf("\n\n");
      }
    }

    if (!reply) {
      return { ok: false, offline: false, message: "The backend sent an empty answer." };
    }
    return { ok: true, data: { reply, tools_used: tools } };
  } catch (err) {
    if (err instanceof Error && err.message) {
      return { ok: false, offline: false, message: err.message };
    }
    return { ok: false, offline: true, message: "The backend is unreachable." };
  } finally {
    clearTimeout(timer);
  }
}

// Returns { ok, offline, message, data }. "offline" means the backend could
// not be reached at all, which is the only case that justifies a demo reply.
async function callBackend(q) {
  const controller = new AbortController();
  const budgetMs = Math.max(
    CHAT_TIMEOUT_FLOOR_MS,
    (serverTimeoutSeconds + CHAT_TIMEOUT_PADDING_S) * 1000,
  );
  const timer = setTimeout(() => controller.abort(), budgetMs);
  try {
    const res = await fetch(backend, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        message: q,
        history: history.slice(-MAX_HISTORY_TURNS),
        model: selectedModel || undefined,
      }),
      signal: controller.signal,
    });

    let payload = null;
    try { payload = await res.json(); } catch { /* not JSON */ }

    if (!res.ok) {
      return { ok: false, offline: false, message: describeError(res.status, payload) };
    }
    if (!payload || typeof payload.reply !== "string") {
      return {
        ok: false,
        offline: false,
        message: "The backend sent a reply this UI could not read.",
      };
    }
    return { ok: true, data: payload };
  } catch (err) {
    if (err && err.name === "AbortError") {
      return {
        ok: false,
        offline: false,
        message: `The backend did not answer within ${Math.round(budgetMs / 1000)}s.`,
      };
    }
    return { ok: false, offline: true, message: "The backend is unreachable." };
  } finally {
    clearTimeout(timer);
  }
}

// --- send ------------------------------------------------------------------
async function send(text) {
  const q = (text ?? input.value).trim();
  if (!q) return;

  if (q.length > MAX_MESSAGE_CHARS) {
    bubble("error", `That message is ${q.length} characters. The backend accepts at most ${MAX_MESSAGE_CHARS}.`);
    return;
  }

  bubble("user", q);
  // Only clear the composer when the message came from it (an example button
  // must not wipe a half-typed message).
  if (text === undefined || text === null) {
    input.value = "";
    updateCounter();
  }
  sendBtn.disabled = true;
  const t = typing();
  let streamBubble = null;

  try {
    if (!demoOnly) {
      // Streamed first: the answer shows up while it is written. If the
      // server has no stream route, fall back to the plain JSON call.
      let result = await callBackendStreaming(q, (text) => {
        if (!streamBubble) {
          // The first token arrived: swap the typing dots for the answer.
          t.remove();
          streamBubble = bubble("agent", "");
        }
        streamBubble.textContent += text;
      });
      if (!result.ok && /stream|empty answer|no stream/i.test(result.message || "")) {
        result = await callBackend(q);
      }

      if (result.ok) {
        if (streamBubble) {
          streamBubble.textContent = result.data.reply || streamBubble.textContent;
          if (result.data.tools_used && result.data.tools_used.length) {
            const meta = document.createElement("span");
            meta.className = "meta";
            result.data.tools_used.forEach((name) => {
              const badge = document.createElement("span");
              badge.className = "badge";
              badge.textContent = "⚙ " + name;
              meta.appendChild(badge);
            });
            streamBubble.appendChild(meta);
          }
        } else {
          bubble("agent", result.data.reply || "(empty)", result.data.tools_used || []);
        }
        recordTurn("user", q);
        recordTurn("assistant", result.data.reply || "");
        if (result.data.error) setStatus("error");
        else if (result.data.demo) setStatus("nokey");
        else setStatus("live");
        return;
      }

      if (!result.offline) {
        // The backend answered with an error: show it instead of faking one.
        bubble("error", result.message);
        setStatus("error");
        return;
      }
      // Backend unreachable: say so, then fall back to the local demo.
      if (hasBackend()) {
        bubble("error", `Backend unreachable at ${backend} - showing a local demo answer.`);
        setStatus("offline");
      } else {
        bubble("error", "No backend is configured - showing a local demo answer.");
        setStatus("static");
      }
    } else {
      t.remove();
    }

    await new Promise((r) => setTimeout(r, 450));
    const d = demoReply(q);
    bubble("agent", d.reply, d.tools);
    setStatus("demo");
  } finally {
    t.remove();
    sendBtn.disabled = false;
    // Do not pop up the on-screen keyboard on touch devices.
    if (window.matchMedia("(pointer: fine)").matches) input.focus();
  }
}

// --- counter ---------------------------------------------------------------
function updateCounter() {
  if (!counter) return;
  const len = input.value.length;
  counter.textContent = len ? `${len}/${MAX_MESSAGE_CHARS}` : "";
  counter.classList.toggle("warn", len > MAX_MESSAGE_CHARS * 0.9);
}

// --- wiring ----------------------------------------------------------------
form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
input.addEventListener("input", updateCounter);
document.querySelectorAll("[data-ex]").forEach((b) =>
  b.addEventListener("click", () => send(b.dataset.ex)));
clearBtn.addEventListener("click", () => { chat.innerHTML = ""; history = []; seed(); });
settingsBtn.addEventListener("click", () => {
  settings.hidden = !settings.hidden;
  if (!settings.hidden) probeHealth();
});

document.getElementById("saveSettings").addEventListener("click", () => {
  backend = normaliseChatUrl(backendUrl.value.trim());
  backendUrl.value = backend;
  demoOnly = false;
  backendChosen = true;
  selectedModel = modelSelect ? modelSelect.value : "";
  if (selectedModel) {
    writeStored(STORAGE_MODEL, selectedModel);
  } else {
    removeStored(STORAGE_MODEL);
  }
  writeStored(STORAGE_BACKEND, backend);
  removeStored(STORAGE_DEMO_ONLY);
  settings.hidden = true;
  bubble(
    "agent",
    `Backend set to ${backend}${selectedModel ? `, model ${selectedModel}` : ""}. Next message will try the live agent.`,
  );
  probeHealth();
});

// Switching the model applies immediately - no need to press Save.
if (modelSelect) {
  modelSelect.addEventListener("change", () => {
    selectedModel = modelSelect.value;
    if (selectedModel) {
      writeStored(STORAGE_MODEL, selectedModel);
    } else {
      removeStored(STORAGE_MODEL);
    }
    bubble(
      "agent",
      selectedModel
        ? `Model set to ${selectedModel}. The next message will use it.`
        : "Model set to automatic (the fastest model available).",
    );
  });
}

document.getElementById("demoBtn").addEventListener("click", () => {
  demoOnly = true;
  writeStored(STORAGE_DEMO_ONLY, "1");
  settings.hidden = true;
  setStatus("demo");
});

resetBtn.addEventListener("click", () => {
  backend = defaultBackend();
  backendUrl.value = backend;
  demoOnly = false;
  backendChosen = false;
  selectedModel = "";
  if (modelSelect) modelSelect.value = "";
  removeStored(STORAGE_BACKEND);
  removeStored(STORAGE_DEMO_ONLY);
  removeStored(STORAGE_MODEL);
  bubble("agent", `Backend reset to ${backend}.`);
  probeHealth();
});

function seed() {
  bubble("agent", "Welcome! I'm your AI assistant. Ask a math question or anything to search.\nType 'quit' in the terminal version — here just keep chatting.");
}

backendUrl.value = backend;
setStatus(demoOnly ? "demo" : "connecting");
seed();
updateCounter();
probeHealth();
