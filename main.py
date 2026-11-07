import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cloner import clone_repo, cleanup_repo
from parser import parse_repo_source
from embedder import index_chunks
from main_test import search
from llm import ask_llm

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

class LoadRequest(BaseModel):
    github_url: str

class ChatRequest(BaseModel):
    question: str
    repo_id: str | None = None

active_repo_id = None
logger = logging.getLogger("repo_chat")

@app.post("/load")
def load_repo(req: LoadRequest):
    global active_repo_id
    logger.info("Starting repository load: %s", req.github_url)
    try:
        path, repo_id = clone_repo(req.github_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    try:
        chunks = parse_repo_source(path, repo_id)
        file_count = len({chunk["file"] for chunk in chunks})
        logger.info("Parsed repository %s: %d files, %d chunks", repo_id, file_count, len(chunks))
        count = index_chunks(chunks, repo_id)
        logger.info("Indexed repository %s: %d chunks", repo_id, count)
        active_repo_id = repo_id
        return {"status": "indexed", "chunks": count, "repo_id": repo_id}
    except (RuntimeError, ValueError) as exc:
        logger.exception("Repository load failed for %s", repo_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        cleanup_repo(path)

@app.post("/chat")
def chat(req: ChatRequest):
    repo_id = req.repo_id or active_repo_id
    if not repo_id:
        raise HTTPException(status_code=400, detail="Load a repository before asking a question.")
    try:
        logger.info("Chat retrieval started: repo=%s question_length=%d", repo_id, len(req.question))
        chunks = search(req.question, top_k=5, repo_id=repo_id)
        logger.info("Chat retrieval returned %d chunks for repo=%s", len(chunks), repo_id)
        answer = ask_llm(req.question, chunks)
        if not answer.strip():
            raise RuntimeError("The AI provider returned an empty answer. Please try again.")
        logger.info("Chat answer generated for repo=%s", repo_id)
    except RuntimeError as exc:
        logger.exception("Chat failed for repo=%s", repo_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"answer": answer, "sources": chunks}