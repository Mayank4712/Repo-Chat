import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath


GITHUB_URL_PATTERN = re.compile(r"^https://github\.com/([^/\s]+)/([^/\s]+?)/?$")
MAX_FILES = 3000
SKIP_DIRS = {".git", "node_modules", "build", "dist", "target", ".venv", "venv", "__pycache__"}
SUPPORTED_EXTENSIONS = {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".md", ".yml", ".yaml", ".json", ".properties", ".gradle"}
SPECIAL_FILES = {"pom.xml", "build.gradle", "package.json", "dockerfile"}
LOCK_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock"}
MAX_FILE_BYTES = 200 * 1024
MAX_CONFIG_BYTES = 100 * 1024


def validate_github_url(github_url: str) -> str:
    match = GITHUB_URL_PATTERN.fullmatch(github_url.strip())
    if not match:
        raise ValueError("Repository URL must use https://github.com/owner/name.")
    return f"https://github.com/{match.group(1)}/{match.group(2)}"


def repository_id(github_url: str) -> str:
    normalized = validate_github_url(github_url)
    return normalized.removeprefix("https://github.com/").rstrip("/")


def _remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _git_command(repo_path, *args):
    return ["git", "-c", "core.longpaths=true", "-C", repo_path, *args]


def _run_git(args, cwd=None):
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _last_git_lines(result):
    output = (result.stderr or result.stdout or "").strip().splitlines()
    return "\n".join(output[-5:]) or "git command failed."


@dataclass
class GitRepository:
    path: str
    repo_id: str

    def list_files(self):
        result = _run_git(_git_command(self.path, "ls-tree", "-r", "-z", "--long", "HEAD"))
        if result.returncode != 0:
            raise RuntimeError(f"Unable to list repository files: {_last_git_lines(result)}")
        entries = []
        for record in result.stdout.split("\0"):
            if not record:
                continue
            metadata, file_path = record.split("\t", 1)
            fields = metadata.split()
            if len(fields) < 4 or fields[1] != "blob":
                continue
            try:
                size = int(fields[3])
            except ValueError:
                continue
            if _should_index_path(file_path, size):
                entries.append((fields[2], file_path, size))
        entries.sort(key=lambda entry: (0 if entry[1].lower() == "readme.md" else 1, 0 if "/" not in entry[1] else 1, entry[1].lower()))
        return entries[:MAX_FILES]

    def read_files(self, entries):
        if not entries:
            return []
        process = subprocess.Popen(
            _git_command(self.path, "cat-file", "--batch"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        request = b"".join(f"{object_id}\n".encode("ascii") for object_id, _, _ in entries)
        stdout, stderr = process.communicate(request)
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
            raise RuntimeError(f"Unable to read repository files: {'\n'.join(detail[-5:]) or 'git cat-file failed.'}")
        position = 0
        files = []
        for object_id, file_path, expected_size in entries:
            header_end = stdout.find(b"\n", position)
            if header_end < 0:
                break
            header = stdout[position:header_end].decode("ascii", errors="replace").split()
            position = header_end + 1
            if len(header) < 3 or header[1] != "blob":
                continue
            size = int(header[2])
            content = stdout[position:position + size]
            position += size + 1
            if size != expected_size:
                continue
            try:
                files.append((file_path, content.decode("utf-8")))
            except UnicodeDecodeError:
                continue
        return files


def _should_index_path(file_path, size):
    path = PurePosixPath(file_path)
    lower_name = path.name.lower()
    if any(part.lower() in SKIP_DIRS for part in path.parts[:-1]):
        return False
    if (
        lower_name in LOCK_NAMES
        or lower_name.startswith(".env")
        or lower_name.endswith((".pem", ".key", ".crt"))
        or "private" in lower_name
        or lower_name.startswith(("id_rsa", "id_ed25519"))
    ):
        return False
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS and lower_name not in SPECIAL_FILES:
        return False
    return size <= (MAX_CONFIG_BYTES if extension in {".json", ".yml", ".yaml"} else MAX_FILE_BYTES)


def clone_repo(github_url: str) -> tuple[GitRepository, str]:
    normalized_url = validate_github_url(github_url)
    temp_root = os.environ.get("CODEBASE_CHAT_CLONE_ROOT", r"D:\gc")
    os.makedirs(temp_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="cb-", dir=temp_root)
    commands = [
        ["git", "-c", "core.longpaths=true", "clone", "--quiet", "--no-checkout", "--filter=blob:none", "--depth", "1", normalized_url, temp_dir],
        ["git", "-c", "core.longpaths=true", "clone", "--quiet", "--no-checkout", "--depth", "1", normalized_url, temp_dir],
    ]
    try:
        for index, command in enumerate(commands):
            result = _run_git(command)
            repository = GitRepository(temp_dir, repository_id(normalized_url))
            if result.returncode == 0:
                if repository.list_files():
                    return repository, repository.repo_id
            elif index == 0 and "filter" in (result.stderr or "").lower():
                cleanup_repo(temp_dir)
                temp_dir = tempfile.mkdtemp(prefix="cb-", dir=temp_root)
                continue
            if index == len(commands) - 1:
                raise RuntimeError(f"Unable to clone repository: {_last_git_lines(result)}")
            cleanup_repo(temp_dir)
            temp_dir = tempfile.mkdtemp(prefix="cb-", dir=temp_root)
        raise RuntimeError("Unable to clone repository: no indexable files found.")
    except Exception:
        cleanup_repo(temp_dir)
        raise


def cleanup_repo(repository) -> None:
    path = repository.path if isinstance(repository, GitRepository) else repository
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=_remove_readonly)
