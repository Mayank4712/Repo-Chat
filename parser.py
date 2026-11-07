import ast
import os
import re
from pathlib import Path


MAX_FILE_BYTES = 200 * 1024
MAX_CONFIG_BYTES = 100 * 1024
MAX_CHARS = 1800
MAX_LINES = 40
SKIP_DIRS = {".git", "node_modules", "build", "dist", "target", ".venv", "venv", "__pycache__"}
SUPPORTED_EXTENSIONS = {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".md", ".yml", ".yaml", ".json", ".properties", ".gradle"}
SPECIAL_FILES = {"pom.xml", "build.gradle", "package.json", "dockerfile"}
LOCK_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock"}


def _split_text(text: str, start_line: int, end_line: int, filepath: str, name: str, class_name=None):
    lines = text.splitlines()
    chunks = []
    for offset in range(0, len(lines), MAX_LINES):
        part = lines[offset:offset + MAX_LINES]
        while len("\n".join(part)) > MAX_CHARS and len(part) > 1:
            part = part[:-1]
        if not part:
            continue
        part_text = "\n".join(part)
        actual_end = start_line + offset + len(part) - 1
        chunks.append({
            "file": filepath,
            "class": class_name,
            "function": name,
            "start_line": start_line + offset,
            "end_line": actual_end,
            "code": part_text,
        })
    return chunks


def _make_chunk(source_lines, start, end, filepath, name, class_name=None):
    return _split_text("\n".join(source_lines[start - 1:end]), start, end, filepath, name, class_name)


def parse_file(filepath: str, display_path: str | None = None, repo_id: str | None = None):
    path = Path(filepath)
    if path.stat().st_size > MAX_FILE_BYTES:
        return []
    shown_path = display_path or str(filepath)
    source = path.read_text(encoding="utf-8", errors="replace")
    return parse_source(source, shown_path)


def parse_source(source: str, filepath: str):
    path = Path(filepath)
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return _split_text(source, 1, len(source.splitlines()), filepath, path.name)
        lines = source.splitlines()
        chunks = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.extend(_make_chunk(lines, child.lineno, child.end_lineno, filepath, child.name, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node in tree.body:
                chunks.extend(_make_chunk(lines, node.lineno, node.end_lineno, filepath, node.name))
        return chunks or _split_text(source, 1, len(lines), filepath, path.name)
    return parse_text_file(source, filepath, path.name)


def _matching_brace(source: str, opening: int) -> int:
    depth = 0
    quote = None
    escaped = False
    for index in range(opening, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def parse_text_file(source: str, filepath: str, filename: str):
    lines = source.splitlines()
    if filename.lower() in {"dockerfile"} or Path(filename).suffix.lower() in {".md", ".yml", ".yaml", ".json", ".properties", ".gradle"}:
        if Path(filename).suffix.lower() == ".md":
            headings = list(re.finditer(r"(?m)^(#{1,6})\s+.+$", source))
            chunks = []
            for index, heading in enumerate(headings):
                start = source[:heading.start()].count("\n") + 1
                end = source[:headings[index + 1].start()].count("\n") if index + 1 < len(headings) else len(lines)
                chunks.extend(_make_chunk(lines, start, max(start, end), filepath, heading.group(0).strip()))
            return chunks or _split_text(source, 1, len(lines), filepath, filename)
        return _split_text(source, 1, len(lines), filepath, filename)

    declarations = []
    for line_number, line_text in enumerate(lines, 1):
        class_match = re.search(
            r"\b(class|interface|enum)\s+([A-Za-z_$][\w$]*)\s*\{", line_text
        )
        function_match = re.search(
            r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)", line_text
        )
        method_match = re.search(
            r"\b([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?::\s*[\w<>\[\]| ]+)?\s*\{",
            line_text,
        )
        if class_match:
            declarations.append((line_number, class_match.group(2), True))
        elif function_match:
            declarations.append((line_number, function_match.group(1), False))
        elif method_match and not line_text.lstrip().startswith(("if ", "for ", "while ")):
            declarations.append((line_number, method_match.group(1), False))
    chunks = []
    completed = []
    for index, (line, name, is_class) in enumerate(declarations):
        line_start_offset = sum(len(item) + 1 for item in lines[:line - 1])
        opening = source.find("{", line_start_offset)
        closing = _matching_brace(source, opening) if opening >= 0 else -1
        end = source[:closing + 1].count("\n") + 1 if closing >= 0 else (
            declarations[index + 1][0] - 1 if index + 1 < len(declarations) else len(lines)
        )
        completed.append((line, end, name, is_class))
    for line, end, name, is_class in completed:
        parent = next(
            (
                class_name
                for class_start, class_end, class_name, class_is_class in completed
                if class_is_class and class_start < line <= class_end
            ),
            None,
        )
        chunks.extend(_make_chunk(lines, line, max(line, end), filepath, name, parent))
    return chunks or _split_text(source, 1, len(lines), filepath, filename)


def parse_repo(folder_path: str, repo_id: str | None = None):
    all_chunks = []
    root = Path(folder_path)
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [directory for directory in dirs if directory not in SKIP_DIRS]
        for filename in files:
            lower_name = filename.lower()
            path = Path(current_root) / filename
            if (
                lower_name in LOCK_NAMES
                or lower_name.startswith(".env")
                or lower_name.endswith((".pem", ".key", ".crt"))
                or "private" in lower_name
                or lower_name.startswith(("id_rsa", "id_ed25519"))
            ):
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS and lower_name not in SPECIAL_FILES:
                continue
            if path.stat().st_size > (MAX_CONFIG_BYTES if path.suffix.lower() in {".json", ".yml", ".yaml"} else MAX_FILE_BYTES):
                continue
            try:
                raw = path.read_bytes()
                raw.decode("utf-8")
                if path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"}:
                    lines = raw.splitlines()
                    if len(lines) < 10 and any(len(line) > 500 for line in lines):
                        continue
                relative = str(path.relative_to(root)).replace(os.sep, "/")
                chunks = parse_file(str(path), relative, repo_id)
                for chunk in chunks:
                    chunk["repo_id"] = repo_id or str(root)
                all_chunks.extend(chunks)
            except (OSError, UnicodeDecodeError):
                continue
    return all_chunks


def parse_repo_source(repository, repo_id: str | None = None):
    all_chunks = []
    for filepath, source in repository.read_files(repository.list_files()):
        chunks = parse_source(source, filepath)
        for chunk in chunks:
            chunk["repo_id"] = repo_id or repository.repo_id
        all_chunks.extend(chunks)
    return all_chunks
