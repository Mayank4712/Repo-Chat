import json
import os
import re

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
API_URL = (os.getenv("BACKEND_URL") or "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 300
GITHUB_URL_PATTERN = re.compile(r"^https://github\.com/([^/\s]+)/([^/\s]+?)/?$")
STARTER_QUESTIONS = (
    "What does this project do?",
    "How is the main functionality organized?",
    "Where should I start reading the code?",
)


st.set_page_config(page_title="Repo Chat", layout="wide")
st.markdown(
    """
<style>
:root {
  --bg: #f6f8fa; --panel: #ffffff; --panel-muted: #f6f8fa; --border: #d0d7de;
  --text: #1f2328; --muted: #656d76; --accent: #0969da; --accent-strong: #0550ae;
  --danger: #cf222e; --success: #1a7f37; --code: #24292f; --radius: 6px;
}
[data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1280px; padding: 24px 28px 40px; }
.topbar, .repo-header, .panel, .message, .empty-state, .error-state {
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
}
.topbar { padding: 16px 20px; margin-bottom: 16px; }
.brand { color: var(--text); font-size: 20px; font-weight: 600; }
.eyebrow { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.repo-header { padding: 18px 20px; margin-bottom: 16px; }
.breadcrumb { color: var(--accent); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 14px; }
.status { display: inline-block; border: 1px solid #a7f3d0; border-radius: 999px; color: var(--success); background: #ecfdf3; padding: 3px 9px; font-size: 12px; }
.panel { padding: 16px; height: 100%; }
.panel-title { font-weight: 600; margin-bottom: 10px; }
.message { padding: 14px 16px; margin-bottom: 12px; }
.message-user { border-left: 3px solid var(--accent); }
.message-assistant { border-left: 3px solid var(--success); }
.message-error { border-left-color: var(--danger); background: #fff1f2; }
.sources { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); }
.source-label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.empty-state, .error-state { padding: 28px; text-align: center; }
.error-state { border-color: #ff8182; color: var(--danger); }
.skeleton { height: 14px; background: #eaeef2; border-radius: 4px; margin: 10px 0; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .45; } }
textarea:focus, input:focus, button:focus { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --panel: #161b22; --panel-muted: #0d1117; --border: #30363d; --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --accent-strong: #79c0ff; }
  .status { background: #12261e; } .error-state { background: #2d1115; } .skeleton { background: #21262d; }
}
</style>
""",
    unsafe_allow_html=True,
)


def backend_error(response):
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        detail = response.text
    return f"Backend error ({response.status_code}): {detail or 'No detail returned.'}"


def validate_url(value):
    if not GITHUB_URL_PATTERN.fullmatch(value.strip()):
        return "Use a public GitHub URL in the form https://github.com/owner/name."
    return None


def initialize_state():
    defaults = {
        "repo_id": None,
        "chunk_count": 0,
        "messages": [],
        "load_error": None,
        "last_question": None,
        "selected_source": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat():
    st.session_state.messages = []
    st.session_state.last_question = None
    st.session_state.selected_source = None


def start_new_repository():
    clear_chat()
    st.session_state.repo_id = None
    st.session_state.chunk_count = 0
    st.session_state.load_error = None


def load_repository(url):
    error = validate_url(url)
    if error:
        st.session_state.load_error = error
        return
    st.session_state.load_error = None
    with st.status("Indexing repository", expanded=True):
        st.markdown("**Reading repository files**")
        st.markdown('<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>', unsafe_allow_html=True)
        try:
            response = requests.post(
                f"{API_URL}/load",
                json={"github_url": url.strip()},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            st.session_state.load_error = f"Backend request failed: {exc}"
            return
        if not response.ok:
            st.session_state.load_error = backend_error(response)
            return
        result = response.json()
        st.session_state.repo_id = result["repo_id"]
        st.session_state.chunk_count = result["chunks"]
        clear_chat()


def ask_question(question):
    question = question.strip()
    if not question or not st.session_state.repo_id:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.last_question = question
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": question, "repo_id": st.session_state.repo_id},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        st.session_state.messages.append({"role": "error", "content": f"Backend request failed: {exc}"})
        return
    if not response.ok:
        st.session_state.messages.append({"role": "error", "content": backend_error(response)})
        return
    try:
        result = response.json()
        answer = (result.get("answer") or "").strip()
        if not answer:
            raise ValueError("The backend returned an empty answer.")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": result.get("sources", [])}
        )
    except (ValueError, TypeError, AttributeError) as exc:
        st.session_state.messages.append({"role": "error", "content": f"Invalid backend response: {exc}"})


initialize_state()

with st.container():
    st.markdown('<div class="topbar"><span class="brand">Repo Chat</span></div>', unsafe_allow_html=True)
    url = st.text_input(
        "Add GitHub repository",
        placeholder="https://github.com/owner/name",
        label_visibility="visible",
    )
    if st.button("Add repository", type="primary", disabled=not url.strip()):
        load_repository(url)
        st.rerun()
    if st.session_state.load_error:
        st.error(st.session_state.load_error)

if not st.session_state.repo_id:
    st.markdown(
        '<div class="empty-state"><h3>Connect a repository to begin</h3><p>Enter a public GitHub URL above, then ask questions about its code.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(
    f'<div class="repo-header"><div class="eyebrow">Repository</div><div class="breadcrumb">{st.session_state.repo_id}</div><span class="status">Ready</span><p>{st.session_state.chunk_count} indexed chunks</p></div>',
    unsafe_allow_html=True,
)

left, center, right = st.columns([1, 2.2, 1.2], gap="medium")
with left:
    st.markdown('<div class="panel"><div class="panel-title">Repository</div>', unsafe_allow_html=True)
    st.caption(st.session_state.repo_id)
    st.caption("Repository metadata, re-indexing, deletion, and file browsing are not exposed by the current API.")
    if st.button("New repository", use_container_width=True):
        start_new_repository()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    st.markdown('<div class="panel"><div class="panel-title">Chat</div>', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown('<div class="empty-state"><h3>Ask about the code</h3><p>Chat history is kept in this browser session only.</p></div>', unsafe_allow_html=True)
        for starter in STARTER_QUESTIONS:
            if st.button(starter, use_container_width=True):
                ask_question(starter)
                st.rerun()
    for index, message in enumerate(st.session_state.messages):
        role = message["role"]
        css = "message-user" if role == "user" else "message-error" if role == "error" else "message-assistant"
        label = "You" if role == "user" else "Error" if role == "error" else "Assistant"
        st.markdown(f'<div class="message {css}"><strong>{label}</strong></div>', unsafe_allow_html=True)
        if role == "assistant":
            st.markdown(message["content"])
            sources = message.get("sources", [])
            if sources:
                st.markdown('<div class="sources"><div class="source-label">Sources</div></div>', unsafe_allow_html=True)
                for source_index, source in enumerate(sources):
                    label = f"{source.get('file', 'source')}:{source.get('start_line', '?')}-{source.get('end_line', '?')}"
                    if st.button(label, key=f"source_{index}_{source_index}", use_container_width=True):
                        st.session_state.selected_source = source
                        st.rerun()
        else:
            st.write(message["content"])
        if role == "error" and st.session_state.last_question and st.button("Retry", key=f"retry_{index}"):
            ask_question(st.session_state.last_question)
            st.rerun()
    question = st.chat_input("Ask a question about this repository")
    if question:
        ask_question(question)
        st.rerun()
    if st.session_state.messages and st.button("Clear chat"):
        clear_chat()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="panel-title">Source details</div>', unsafe_allow_html=True)
    source = st.session_state.selected_source
    if not source:
        st.caption("Select a source chip to inspect the cited code.")
    else:
        st.markdown(f"**{source.get('file', 'source')}**")
        st.caption(f"Lines {source.get('start_line', '?')}-{source.get('end_line', '?')}")
        st.code(source.get("code", ""), language="python")
        if st.button("Close source"):
            st.session_state.selected_source = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
