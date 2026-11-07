from cloner import clone_repo, cleanup_repo
from parser import parse_repo_source

repo_url = "https://github.com/karpathy/micrograd"

print("Cloning repo...")
path, repo_id = clone_repo(repo_url)
try:
    print(f"Cloned to: {path}")
    print("Parsing repo...")
    chunks = parse_repo_source(path, repo_id)
    print(f"\nFound {len(chunks)} chunks total.\n")

    for chunk in chunks[:10]:
        label = f"{chunk['class']}.{chunk['function']}" if chunk['class'] else chunk['function']
        print(f"--- {label} ({chunk['file']}, lines {chunk['start_line']}-{chunk['end_line']}) ---")
finally:
    cleanup_repo(path)
    print("\nCleaned up temp folder.")