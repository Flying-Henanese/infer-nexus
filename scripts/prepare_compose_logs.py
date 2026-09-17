#!/usr/bin/env python3
"""Prepare host log mounts and persist their absolute root in the Compose .env."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICES = ("ray-head", "ray-worker", "serve-deployer")


def read_value(text: str, key: str, default: str = "") -> str:
    value = default
    for line in text.splitlines():
        match = re.match(rf"^\s*(?:export\s+)?{key}\s*=(.*)$", line)
        if match:
            tokens = shlex.split(match[1], comments=True)
            if len(tokens) > 1:
                raise ValueError(f"Quote values containing spaces: {key}")
            value = tokens[0] if tokens else ""
    return value


def prepare(env_file: Path, logs_dir: str | None) -> Path:
    env_file = env_file.resolve()
    text = env_file.read_text() if env_file.exists() else (ROOT / ".env.template").read_text()
    configured = logs_dir or read_value(text, "LOGS_HOST_PATH")
    log_root = (
        Path(configured)
        if configured
        else env_file.parent.parent / f"{env_file.parent.name}-logs"
    )
    if not log_root.is_absolute() or any(c in str(log_root) for c in "$\n\r'"):
        raise ValueError("LOGS_HOST_PATH must be a literal absolute path (no variable expansion)")
    log_root = log_root.resolve()
    if log_root == Path("/") or log_root == env_file.parent:
        raise ValueError("Use a dedicated log directory, not / or the repository root")
    uid = int(read_value(text, "APP_UID", "10001"))
    gid = int(read_value(text, "APP_GID", "10001"))
    if uid <= 0 or gid <= 0:
        raise ValueError("APP_UID and APP_GID must be positive non-root IDs")
    for service in SERVICES:
        directory = log_root / service
        ray_directory = directory / "ray"
        if directory.is_symlink() or ray_directory.is_symlink():
            raise ValueError(f"Service log directories must not be symlinks: {directory}")
        # Never recursively chown existing logs. Existing files require explicit migration.
        for path in (directory, ray_directory):
            if path.exists():
                if not path.is_dir():
                    raise ValueError(f"Expected a log directory: {path}")
                stat = path.stat()
                if (stat.st_uid, stat.st_gid, stat.st_mode & 0o777) == (uid, gid, 0o755):
                    continue
            command = ["install", "-d", "-m", "0755", "-o", str(uid), "-g", str(gid), str(path)]
            ancestor = path
            while not ancestor.exists():
                ancestor = ancestor.parent
            needs_sudo = (
                uid != os.geteuid()
                or gid != os.getegid()
                or path.exists() and path.stat().st_uid != os.geteuid()
                or not os.access(ancestor, os.W_OK | os.X_OK)
            )
            if os.geteuid() != 0 and needs_sudo:
                command.insert(0, "sudo")
            print(f"Preparing {path} for {uid}:{gid}", flush=True)
            subprocess.run(command, check=True)
    assignment = f"LOGS_HOST_PATH='{log_root}'"
    pattern = r"(?m)^[ \t]*(?:export[ \t]+)?LOGS_HOST_PATH[ \t]*=.*$"
    if re.search(pattern, text):
        text = re.sub(pattern, lambda _: assignment, text)
    else:
        text = text.rstrip() + "\n" + assignment + "\n"
    env_file.write_text(text)
    print(f"Configured {env_file}: LOGS_HOST_PATH={log_root}")
    print("Existing log files are preserved; their ownership is not changed.")
    return log_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--logs-dir", help="Override the configured log root with an absolute path")
    args = parser.parse_args()
    try:
        prepare(args.env_file, args.logs_dir)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Log preparation failed: {error}\n")


if __name__ == "__main__":
    main()
