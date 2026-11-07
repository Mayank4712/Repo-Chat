import subprocess

from cloner import GitRepository, cleanup_repo
from parser import parse_repo_source


def _run_git(*args):
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_checkout_free_reader_handles_windows_problem_paths(tmp_path):
    repository_path = tmp_path / "source"
    _run_git("init", str(repository_path))
    long_path = "/".join(["long"] * 70) + "/module.py"
    normal_source = "def normal():\n    return True\n"
    reserved_source = "not a python file\n"
    long_source = "def long_file():\n    return True\n"
    (repository_path / "normal.py").write_text(normal_source, encoding="utf-8")
    _run_git("-C", str(repository_path), "add", "normal.py")
    _run_git("-C", str(repository_path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial")
    hashes = {}
    for path, content in (("CON", reserved_source), (long_path, long_source)):
        result = subprocess.run(
            ["git", "-C", str(repository_path), "hash-object", "-w", "--stdin"],
            input=content,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        hashes[path] = result.stdout.strip()
    index_info = "".join(f"100644 {object_id}\t{path}\n" for path, object_id in hashes.items())
    subprocess.run(
        ["git", "-C", str(repository_path), "update-index", "--add", "--index-info"],
        input=index_info,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    tree = _run_git("-C", str(repository_path), "write-tree").stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(repository_path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit-tree", tree, "-p", "HEAD"],
        input="test\n",
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    _run_git("-C", str(repository_path), "update-ref", "refs/heads/master", commit)

    repository = GitRepository(str(repository_path), "local/test")
    try:
        entries = repository.list_files()
        files = repository.read_files(entries)
        chunks = parse_repo_source(repository, "local/test")
    finally:
        cleanup_repo(repository_path)

    assert "normal.py" in {path for path, _ in files}
    assert any(chunk["file"] == "normal.py" for chunk in chunks)
