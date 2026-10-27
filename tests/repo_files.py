"""Source inventories for repository-wide checks, excluding local nested copies."""

from pathlib import Path
from subprocess import check_output


def python_files(root: Path):
    paths = check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=root).decode().split("\0")
    return [root / name for name in paths if name and (root / name).is_file()]
