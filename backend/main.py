"""Repo Chat backend — FastAPI service.

Loads a public GitHub repository, chunks and indexes its text files,
and answers questions about the code with file/line citations.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
from typing import List, Optional

import chromadb
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("repochat")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
PORT = int(os.getenv("PORT", "8000"))

MAX_FILES = 200
MAX_FILE_BYTES = 200_000
CHUNK_LINES = 40
CHUNK_OVERLAP = 10
TOP_K_VECTOR = 8
TOP_K_KEYWORD = 8

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cs", ".swift",
    ".m", ".mm", ".scala", ".clj", ".hs", ".ml", ".fs", ".erl", ".ex", ".exs",
    ".lua", ".r", ".R", ".pl", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".sql", ".html", ".htm", ".css", ".scss", ".sass", ".less", ".xml",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".md",
    ".rst", ".txt", ".dockerfile", ".gitignore", ".env.example",
}
SKIP_NAMES = {
    ".gitignore", ".gitattributes", ".dockerignore", "license", "readme.md",
    "readme.rst", "readme.txt",
}
SKIP_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", ".venv", "venv",
    "build", "dist", "target", ".next", "out",
}

REPO_URL_RE = re.compile(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/?#]+?)(?:\.git)?/?$")


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class LoadRepoRequest(BaseModel):
    repo_url: str


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    file: str
    start_line: int
    end_line: int


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class StatusResponse(BaseModel):
    loaded: bool
    repo_url: Optional[str] = None
    indexed_count: int = 0
    conversation_length: int = 0


class Message(BaseModel):
    role: str
    content: str


# --------------------------------------------------------------------------- #
# GitHub client
# --------------------------------------------------------------------------- #

class GitHubClient:
    def __init__(self, token: str = "") -> None:
        self.token = token
        self.session = requests.Session()
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "repochat"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def _handle(self, response: requests.Response, context: str) -> None:
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="GitHub authentication failed.")
        if response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="GitHub rate limit exceeded or repository not accessible.",
            )
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Repository not found.")
        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"GitHub API error while {context}.",
            )

    def parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        match = REPO_URL_RE.search(repo_url.strip())
        if not match:
            raise HTTPException(
                status_code=400,
                detail="Invalid repository address. Use a GitHub URL like https://github.com/owner/repo",
            )
        return match.group("owner"), match.group("repo")

    def default_branch(self, owner: str, repo: str) -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = self.session.get(url, timeout=30)
        self._handle(response, "fetching repository info")
        return response.json()["default_branch"]

    def list_files(self, owner: str, repo: str, ref: str) -> List[dict]:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
        response = self.session.get(url, timeout=60)
        self._handle(response, "listing repository files")
        data = response.json()
        if data.get("truncated"):
            log.warning("GitHub tree response was truncated")
        return [t for t in data.get("tree", []) if t.get("type") == "blob"]

    def file_content(self, owner: str, repo: str, path: str, ref: str) -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
        response = self.session.get(url, timeout=30)
        self._handle(response, f"fetching file {path}")
        data = response.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return data.get("content", "")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def is_indexable(path: str, size: int) -> bool:
    lower = path.lower()
    parts = [p for p in lower.split("/") if p]
    if any(part in SKIP_DIRS for part in parts):
        return False
    name = parts[-1] if parts else lower
    if name in SKIP_NAMES:
        return False
    if size > MAX_FILE_BYTES:
        return False
    _, ext = os.path.splitext(name)
    return ext.lower() in TEXT_EXTENSIONS or name in {
        "makefile", "dockerfile", "gemfile", "rakefile",
    }


def chunk_text(text: str, file_path: str) -> List[dict]:
    lines = text.split("\n")
    chunks: List[dict] = []
    i = 0
    while i < len(lines):
        end = min(i + CHUNK_LINES, len(lines))
        chunk_lines = lines[i:end]
        chunk_text_blob = "\n".join(chunk_lines)
        if chunk_text_blob.strip():
            chunks.append(
                {
                    "file": file_path,
                    "start_line": i + 1,
                    "end_line": end,
                    "text": chunk_text_blob,
                }
            )
        if end >= len(lines):
            break
        i = end - CHUNK_OVERLAP
        if i < 0:
            i = 0
    return chunks


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

class Session:
    def __init__(self) -> None:
        self.repo_url: Optional[str] = None
        self.repo_id: Optional[str] = None
        self.indexed_count: int = 0
        self.chunks: List[dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.conversation: List[Message] = []
        self._chroma_collection = None

    def reset(self) -> None:
        self.repo_url = None
        self.repo_id = None
        self.indexed_count = 0
        self.chunks = []
        self.bm25 = None
        self.conversation = []
        self._chroma_collection = None


session = Session()

# Lazy singletons
_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_llm_client: Optional[OpenAI] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info("Loading embedding model %s", EMBEDDING_MODEL_NAME)
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_llm() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        if not OPENROUTER_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenRouter API key is not configured on the server.",
            )
        _llm_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _llm_client


def repo_id_for(owner: str, repo: str) -> str:
    raw = f"{owner}/{repo}".lower()
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    safe = re.sub(r"[^a-z0-9_-]", "_", repo)
    return f"repo_{safe[:40]}_{digest}"


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(title="Repo Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        loaded=session.repo_url is not None,
        repo_url=session.repo_url,
        indexed_count=session.indexed_count,
        conversation_length=len(session.conversation),
    )


@app.post("/api/load-repo")
def load_repo(request: LoadRepoRequest) -> dict:
    github = GitHubClient(GITHUB_TOKEN)
    owner, repo = github.parse_repo_url(request.repo_url)

    try:
        ref = github.default_branch(owner, repo)
        trees = github.list_files(owner, repo, ref)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network safety net
        log.exception("GitHub error")
        raise HTTPException(status_code=502, detail="Unable to reach GitHub.")

    # Filter + fetch indexable files
    chunks: List[dict] = []
    seen_paths: set[str] = set()
    fetched = 0

    for entry in trees:
        if fetched >= MAX_FILES:
            break
        path = entry.get("path", "")
        if not is_indexable(path, entry.get("size", 0) or 0):
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        fetched += 1

        try:
            content = github.file_content(owner, repo, path, ref)
        except HTTPException:
            continue
        except Exception:
            log.warning("Failed to fetch %s", path)
            continue

        if not content.strip():
            continue
        chunks.extend(chunk_text(content, path))

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No indexable text files found in the repository.",
        )

    # Truncate to a sane total chunk count
    chunks = chunks[:MAX_FILES * 60]

    # Embed
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # Chroma collection (fresh for this repo)
    client = get_chroma()
    collection_name = repo_id_for(owner, repo)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    ids = [f"{i}" for i in range(len(chunks))]
    metadatas = [{"file": c["file"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in chunks]
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())

    # BM25
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Reset session state for new repo
    session.reset()
    session.repo_url = request.repo_url
    session.repo_id = collection_name
    session.indexed_count = len(chunks)
    session.chunks = chunks
    session.bm25 = bm25
    session._chroma_collection = collection

    log.info("Indexed %d chunks from %s", len(chunks), request.repo_url)
    return {
        "message": "Repository indexed.",
        "indexed_count": len(chunks),
        "repo_url": request.repo_url,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if session.repo_id is None or session._chroma_collection is None:
        raise HTTPException(status_code=400, detail="No repository is loaded. Load one first.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    client = get_chroma()
    collection = client.get_collection(name=session.repo_id)
    embedder = get_embedder()

    # Vector search
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    vres = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=TOP_K_VECTOR,
        include=["documents", "metadatas", "distances"],
    )

    # Keyword search
    bm25_scores = session.bm25.get_scores(question.split()) if session.bm25 else []
    kw_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_KEYWORD]

    # Combine: gather candidates
    candidates: dict[int, dict] = {}
    vdocs = vres.get("documents", [[]])[0] if vres.get("documents") else []
    vmetas = vres.get("metadatas", [[]])[0] if vres.get("metadatas") else []
    vdists = vres.get("distances", [[]])[0] if vres.get("distances") else []
    for rank, (doc, meta, dist) in enumerate(zip(vdocs, vmetas, vdists)):
        candidates[rank] = {
            "text": doc,
            "file": meta["file"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "vector_rank": rank,
            "keyword_rank": None,
        }

    for rank, idx in enumerate(kw_order):
        c = session.chunks[idx]
        key = f"kw{idx}"
        candidates[key] = {
            "text": c["text"],
            "file": c["file"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "vector_rank": None,
            "keyword_rank": rank,
        }

    # Dedup by (file, start_line)
    seen: set[tuple[str, int]] = set()
    selected: List[dict] = []
    ordered = sorted(
        candidates.values(),
        key=lambda c: (0 if c["vector_rank"] is not None else 1,
                       c["vector_rank"] if c["vector_rank"] is not None else 999,
                       c["keyword_rank"] if c["keyword_rank"] is not None else 999),
    )
    for c in ordered:
        k = (c["file"], c["start_line"])
        if k in seen:
            continue
        seen.add(k)
        selected.append(c)
        if len(selected) >= 6:
            break

    if not selected:
        raise HTTPException(status_code=424, detail="No relevant code found for the question.")

    # Build context
    context_parts = []
    citations: List[Citation] = []
    for c in selected:
        cite = f"{c['file']}:{c['start_line']}-{c['end_line']}"
        context_parts.append(f"[{cite}]\n{c['text']}")
        citations.append(Citation(file=c["file"], start_line=c["start_line"], end_line=c["end_line"]))
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful coding assistant. Answer the user's question using only "
        "the provided code snippets from a single repository. Each snippet is tagged "
        "with [file:start_line-end_line]. Write a plain-language answer and cite the "
        "specific files and line ranges you relied on using the same tags. If the "
        "snippets do not contain enough information, say so honestly. Do not invent "
        "file names or line numbers."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Code snippets:\n{context}\n\n"
        f"Answer with citations like [src/utils.py:10-25]."
    )

    try:
        llm = get_llm()
        completion = llm.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content or ""
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("LLM error")
        msg = str(exc)
        if "rate limit" in msg.lower() or "429" in msg:
            detail = "The language model is rate-limited. Please try again shortly."
        elif "401" in msg or "authentication" in msg.lower() or "api key" in msg.lower():
            detail = "Language model authentication failed."
        else:
            detail = "The language model returned an error."
        raise HTTPException(status_code=502, detail=detail)

    if not answer.strip():
        raise HTTPException(status_code=502, detail="The language model returned an empty answer.")

    session.conversation.append(Message(role="user", content=question))
    session.conversation.append(Message(role="assistant", content=answer))

    return AskResponse(answer=answer, citations=citations)


@app.post("/api/reset")
def reset() -> dict:
    session.reset()
    return {"message": "Session cleared."}


@app.get("/api/history", response_model=List[Message])
def history() -> List[Message]:
    return list(session.conversation)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
