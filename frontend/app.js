const API = "/api";
const statusEl = document.getElementById("status");
const loadBtn = document.getElementById("loadBtn");
const resetBtn = document.getElementById("resetBtn");
const loadMsg = document.getElementById("loadMsg");
const resetMsg = document.getElementById("resetMsg");
const repoUrlInput = document.getElementById("repoUrl");
const messagesEl = document.getElementById("messages");
const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");

let loaded = false;

async function setStatus() {
  try {
    const res = await fetch(`${API}/status`);
    const st = await res.json();
    loaded = st.loaded;
    statusEl.textContent = st.loaded
      ? `Loaded: ${st.indexed_count} pieces indexed`
      : "No repository loaded";
    questionInput.disabled = !st.loaded;
    sendBtn.disabled = !st.loaded;
  } catch (e) {
    statusEl.textContent = "Status unavailable";
  }
}

function show(el, text, kind) {
  el.textContent = text;
  el.className = "msg " + kind;
  setTimeout(() => { el.textContent = ""; }, 5000);
}

function appendMessage(role, text, citations) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  div.appendChild(bubble);
  if (citations && citations.length) {
    const cites = document.createElement("div");
    cites.className = "citations";
    cites.textContent = "Sources: ";
    citations.forEach((c, i) => {
      const span = document.createElement("span");
      span.className = "cite";
      span.textContent = `${i ? "; " : ""}${c.file}:${c.start_line}-${c.end_line}`;
      cites.appendChild(span);
    });
    div.appendChild(cites);
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

loadBtn.addEventListener("click", async () => {
  const url = repoUrlInput.value.trim();
  if (!url) {
    show(loadMsg, "Please enter a repository URL.", "err");
    return;
  }
  loadBtn.disabled = true;
  loadBtn.textContent = "Indexing...";
  try {
    const res = await fetch(`${API}/load-repo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_url: url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to load repository.");
    show(loadMsg, `Indexed ${data.indexed_count} pieces.`, "ok");
    messagesEl.innerHTML = "";
    appendMessage("assistant", `Repository loaded. I indexed ${data.indexed_count} pieces of code. Ask me a question about it.`);
    await setStatus();
  } catch (e) {
    show(loadMsg, e.message, "err");
  } finally {
    loadBtn.disabled = false;
    loadBtn.textContent = "Load & index";
  }
});

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  try {
    await fetch(`${API}/reset`, { method: "POST" });
    messagesEl.innerHTML = "";
    show(resetMsg, "Session cleared.", "ok");
    await setStatus();
  } catch (e) {
    show(resetMsg, "Failed to clear session.", "err");
  } finally {
    resetBtn.disabled = false;
  }
});

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = questionInput.value.trim();
  if (!q) return;
  questionInput.disabled = true;
  sendBtn.disabled = true;
  appendMessage("user", q);
  questionInput.value = "";
  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to get an answer.");
    appendMessage("assistant", data.answer, data.citations);
  } catch (err) {
    appendMessage("assistant", `Error: ${err.message}`);
  } finally {
    questionInput.disabled = !loaded;
    sendBtn.disabled = !loaded;
  }
});

setStatus();
