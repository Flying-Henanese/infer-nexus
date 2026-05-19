"""Run only the multimodal contract tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_multimodal_contract.py",
        "tests/test_api.py",
        "tests/test_runtime.py",
        "-k",
        "multimodal or vision_model or text_only_model",
        "-rX",
    ]
    completed = subprocess.run(command, cwd=repo_root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
