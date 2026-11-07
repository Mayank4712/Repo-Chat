# Repo Chat

Chat with any public GitHub repository. Paste a repository URL, index its code, and ask questions with cited sources.

Repo Chat is a code-assistant application that clones a public GitHub repository, parses its source files into chunks, stores the chunks in a Qdrant vector database, and answers your questions using Google Gemini — citing the exact files and line ranges it used.

## Features

- Load and index any public GitHub repository by URL
- Parse source code into function- and class-level chunks (Python, JavaScript, TypeScript, Java, Markdown, YAML, JSON, and more)
- Embed chunks with the BAAI/bge-small-en-v1.5 sentence-transformers model
- Store and search chunks in Qdrant (cloud vector database)
- Hybrid retrieval: semantic search combined with keyword search (reciprocal rank fusion)
- Ask questions about a repository and get grounded answers with file:line citations
- Streamlit web frontend and FastAPI backend
- Deployable to Vercel (serverless Python) via vercel.json

## How it works

```
You (browser)
  │
  ▼
Streamlit frontend (app.py)  http://localhost:8501
  │  sends HTTP requests to
  ▼
FastAPI backend (main.py)    http://localhost:8000
  ├─ POST /load
  │   ├─ cloner.py  clones the public GitHub repo to a temp folder
  │   ├─ parser.py  parses source files into chunks
  │   └─ embedder.py embeds chunks and upserts them into Qdrant
  └─ POST /chat
      ├─ main_test.py  hybrid semantic + keyword search over Qdrant
      └─ llm.py        asks Google Gemini with retrieved code as context
```

1. You paste a public GitHub URL into the Streamlit UI.
2. The backend clones the repository, parses supported source files into chunks, embeds each chunk, and stores the vectors in Qdrant tagged with a repository ID.
3. You ask a question. The backend retrieves the most relevant chunks (semantic + keyword, fused by reciprocal rank fusion) and sends them with your question to Gemini.
4. Gemini answers using only the retrieved code and cites file paths and line ranges.

## Tech stack

- Language: Python 3.12 (tested; 3.10+ should work)
- Backend framework: FastAPI
- Frontend: Streamlit (pure Python; no Node.js build step)
- LLM client: google-genai (Google Gemini)
- Embeddings: sentence-transformers (BAAI/bge-small-en-v1.5)
- Vector database: qdrant-client (Qdrant Cloud)
- Keyword search: rank-bm25
- ASGI server: uvicorn
- Env loading: python-dotenv
- HTTP client (frontend → backend): requests / httpx

## Project structure

```
Repo-Chat/
├── .env.example            # Template for environment variables (copy to .env)
├── .gitignore              # Files ignored by git
├── LICENSE                 # MIT License
├── README.md               # This file
├── app.py                  # Streamlit frontend (chat UI)
├── cloner.py               # Git clone + repository file listing
├── embedder.py             # Qdrant client + sentence-transformers indexing
├── eval.py                 # Retrieval evaluation script
├── index_repo.py           # One-off script to index a repo into Qdrant
├── llm.py                  # Google Gemini client (grounded Q&A)
├── main.py                 # FastAPI backend (/load, /chat)
├── main_test.py            # Hybrid semantic + keyword search (RRF)
├── parser.py               # Source-code parser/chunker
├── requirements.txt        # Python dependencies (pinned)
├── sample.py               # Sample code file
├── search_test.py          # Local Qdrant search test
├── test_repo_parse.py      # Repository parsing test
├── vercel.json             # Vercel serverless Python config
├── api/
│   └── index.py            # Vercel entry point (imports main.app)
└── tests/
    ├── test_git_source.py
    ├── test_llm_config.py
    ├── test_parser_languages.py
    └── fixtures/
        ├── Sample.java
        └── sample.ts
```

## Prerequisites

| Tool | Minimum version | Download | Check command |
|------|-----------------|----------|---------------|
| Python | 3.12 | https://www.python.org/downloads/ | `python --version` |
| Git | 2.x | https://git-scm.com/downloads | `git --version` |

No Node.js, npm, or package manager is required. The frontend is Streamlit, which is installed as a Python package from `requirements.txt`. There is no `package.json` or lock file in this project.

## Getting the project

Option 1 — git clone:
```
git clone https://github.com/Mayank4712/Repo-Chat.git
cd Repo-Chat
```

Option 2 — ZIP download:
1. Go to https://github.com/Mayank4712/Repo-Chat
2. Click Code → Download ZIP
3. Extract and open the folder in a terminal

## Setup

### Windows (PowerShell)

1. Create the virtual environment:
```
python -m venv .venv
```

2. Activate it:
```
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Create `.env` from the template:
```
copy .env.example .env
```

5. Edit `.env` and fill in the real values (see table below).

### Windows (CMD)

1. Create the virtual environment:
```
python -m venv .venv
```

2. Activate it:
```
.venv\Scripts\activate.bat
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Create `.env`:
```
copy .env.example .env
```

5. Edit `.env` and fill in the real values.

### macOS / Linux

1. Create the virtual environment:
```
python3 -m venv .venv
```

2. Activate it:
```
source .venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Create `.env`:
```
cp .env.example .env
```

5. Edit `.env` and fill in the real values.

### Environment variables

Copy `.env.example` to `.env`, then replace each placeholder with a real value.

| Variable | Purpose | Required? | Example value | Where to get it |
|----------|---------|-----------|---------------|-----------------|
| `GEMINI_API_KEY` | API key for Google Gemini (used by llm.py) | Required | `replace_with_gemini_api_key` | Google AI Studio (https://aistudio.google.com — free tier available) |
| `GEMINI_MODEL` | Gemini model name | Optional (default `gemini-1.5-flash`) | `gemini-1.5-flash` | Google AI Studio |
| `QDRANT_URL` | HTTPS endpoint of your Qdrant Cloud cluster (used by embedder.py) | Required | `https://your-cluster-id.region.aws.cloud.qdrant.io` | Qdrant Cloud (https://cloud.qdrant.io — free tier available) |
| `QDRANT_API_KEY` | Qdrant Cloud API key with read/write access | Required | `replace_with_qdrant_api_key` | Qdrant Cloud dashboard |
| `BACKEND_URL` | FastAPI backend URL used by the Streamlit frontend | Optional (default `http://localhost:8000`) | `http://localhost:8000` | — |

### Model, database, and vector-store setup

- Embedding model `BAAI/bge-small-en-v1.5` is downloaded automatically the first time `embedder.py` runs (sentence-transformers). No manual download is needed.
- Vector store: Qdrant Cloud. Create a free cluster at https://cloud.qdrant.io and copy its HTTPS endpoint and API key into `QDRANT_URL` and `QDRANT_API_KEY`. No local vector-store folder is required by the app.
- The app clones repositories into a temporary folder (configured by `CODEBASE_CHAT_CLONE_ROOT`, default `D:\gc` on Windows) and removes them after indexing. No persistent local data folder is required to start the app.

## Running the app

Open two terminals in the project folder, activate the virtual environment in both, then:

Terminal 1 — Backend (FastAPI):
```
uvicorn main:app --reload --port 8000
```
Success: you see `Uvicorn running on http://127.0.0.1:8000` (or `http://0.0.0.0:8000`).

Terminal 2 — Frontend (Streamlit):
```
streamlit run app.py
```
Success: Streamlit opens a browser tab at `http://localhost:8501`.

Open `http://localhost:8501` in your browser. Paste a public GitHub URL, click **Add repository**, wait for indexing to finish, then ask a question.

## How to use it

1. Start the backend and frontend as shown above.
2. Open `http://localhost:8501`.
3. In **Add GitHub repository**, paste a public GitHub URL, e.g. `https://github.com/owner/name`.
4. Click **Add repository**. The UI shows **Indexing repository**, then **Ready** with the chunk count.
5. Ask a question in the chat input, e.g.:
   - "What does this project do?"
   - "How is the main functionality organized?"
   - "Where should I start reading the code?"
6. Read the answer. For assistant messages with sources, click a **Sources** chip (file:line) to inspect the cited code in the **Source details** panel.
7. To load a different repository, click **New repository**.

## API endpoints

| Method | Path | Purpose | Body |
|--------|------|---------|------|
| `POST` | `/load` | Clone and index a public GitHub repository | `{"github_url": "https://github.com/owner/name"}` |
| `POST` | `/chat` | Ask a question about an indexed repository | `{"question": "how does login work", "repo_id": "owner/name"}` |

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `GEMINI_API_KEY must be set` | Missing or invalid Gemini key | Set `GEMINI_API_KEY` in `.env` from Google AI Studio, then restart the backend |
| Qdrant connection error | Missing/invalid Qdrant URL or API key, or cluster unreachable | Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` from Qdrant Cloud |
| Port 8000 or 8501 already in use | Another process is using the port | Stop the other process, or change the port (`uvicorn ... --port 8001`, Streamlit `--server.port 8502`) |
| Venv not activating (PowerShell) | Execution policy blocks scripts | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell, then retry activation |
| `ModuleNotFoundError` | Dependency not installed or venv not activated | Activate `.venv` and run `pip install -r requirements.txt` |
| CORS errors | Frontend and backend on different origins without CORS | The backend does not configure CORS; keep `BACKEND_URL` matching the backend host/port |
| GitHub rate limit | Cloning many public repos without auth | Wait and retry; this project clones anonymously and does not support a `GITHUB_TOKEN` |
| Only private repos | `git clone` fails with 404/auth | This project only supports public GitHub repositories |

## Limitations (from the code)

- Only public GitHub repositories are supported (anonymous `git clone`; no `GITHUB_TOKEN` option in the code).
- Repository cloning uses a temporary folder (default `D:\gc` on Windows via `CODEBASE_CHAT_CLONE_ROOT`); ensure that path is writable.
- A maximum of 3000 files per repository are indexed (`MAX_FILES` in cloner.py).
- Individual files larger than 200 KB are skipped (config files larger than 100 KB are skipped).
- Lock files (package-lock.json, yarn.lock, etc.), `.env`-prefixed files, `.pem`, `.key`, `.crt`, and files with "private" in the name are never indexed.
- Full functionality requires a working Gemini API key and a reachable Qdrant Cloud cluster; without them, the UI loads but chat and indexing will fail.
- The frontend (`app.py`) and backend (`main.py`) must both be running; `app.py` calls `BACKEND_URL` (default `http://localhost:8000`).

## Contributing

Contributions are welcome. Open an issue or submit a pull request on GitHub. Please keep the app logic unchanged and report bugs instead of fixing them silently.

## License

MIT License. See `LICENSE` for details.

## Author

Mayank4712
