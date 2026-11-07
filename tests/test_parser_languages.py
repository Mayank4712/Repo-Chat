from pathlib import Path

from parser import parse_file, parse_repo


FIXTURES = Path(__file__).parent / "fixtures"


def test_java_methods_include_line_metadata():
    chunks = parse_file(str(FIXTURES / "Sample.java"))
    names = {chunk["function"] for chunk in chunks}
    assert "Sample" in names
    assert "add" in names
    assert all(chunk["start_line"] <= chunk["end_line"] for chunk in chunks)


def test_typescript_interface_and_function_are_parsed():
    chunks = parse_file(str(FIXTURES / "sample.ts"))
    names = {chunk["function"] for chunk in chunks}
    assert {"User", "greet"} <= names


def test_repository_assigns_repo_identifier():
    chunks = parse_repo(str(FIXTURES), "owner/repo")
    assert chunks
    assert all(chunk["repo_id"] == "owner/repo" for chunk in chunks)
