from cloner import clone_repo, cleanup_repo
from parser import parse_repo_source
from embedder import index_chunks

repo_url = "https://github.com/karpathy/micrograd"

print("Cloning repo...")
path, repo_id = clone_repo(repo_url)
try:
    print("Parsing repo...")
    chunks = parse_repo_source(path, repo_id)
    print(f"Found {len(chunks)} chunks.")

    print("Embedding and storing in Qdrant...")
    count = index_chunks(chunks, repo_id)
    print(f"Stored {count} chunks in Qdrant.")
finally:
    cleanup_repo(path)
    print("Done. Temp folder cleaned up.")