"""Local, read-only queries over infer-nexus event files and Ray sessions.

The query layer deliberately keeps application events and infrastructure text as
different source types.  It never edits, merges, or deletes the files it reads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any, Iterable, Sequence


SERVICE_NAMES = ("ray-head", "ray-worker", "serve-deployer")
_DURATION_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smhd])$")
_TIMESTAMP_RE = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))"
)
_IMPORTANT_EVENTS = (
    "failed",
    "rejected",
    "timeout",
    "cancelled",
    "initializing",
    "starting",
    "ready",
    "shutdown",
    "opened",
    "closed",
)


class LogQueryError(ValueError):
    """Raised when a query root or filter is unsafe or unusable."""


@dataclass(frozen=True, slots=True)
class ApplicationRecord:
    """One parsed JSONL application event with its physical source location."""

    fields: dict[str, Any]
    path: Path
    line_number: int
    service: str
    raw_line: str = ""

    @property
    def timestamp(self) -> datetime | None:
        return parse_timestamp(self.fields.get("timestamp"))


@dataclass(frozen=True, slots=True)
class RawLine:
    """One infrastructure or unsupported-format line with source attribution."""

    path: Path
    line_number: int
    service: str
    text: str
    timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class FileStats:
    """Count and byte total for one retained source category."""

    files: int = 0
    bytes: int = 0
    oldest: datetime | None = None
    newest: datetime | None = None

    def add(self, size: int, modified_at: datetime | None = None) -> "FileStats":
        if modified_at is not None:
            modified_at = modified_at.astimezone(timezone.utc)
        oldest = self.oldest
        newest = self.newest
        if modified_at is not None:
            oldest = modified_at if oldest is None else min(oldest, modified_at)
            newest = modified_at if newest is None else max(newest, modified_at)
        return FileStats(
            files=self.files + 1,
            bytes=self.bytes + size,
            oldest=oldest,
            newest=newest,
        )


@dataclass(frozen=True, slots=True)
class LogStats:
    """Storage coverage report with no deletion or health inference."""

    events: FileStats
    ray_logs: FileStats
    ray_other: FileStats
    historical_sessions: FileStats


@dataclass(frozen=True, slots=True)
class QueryFilters:
    """Safe filters shared by finite and follow queries."""

    since: datetime | None = None
    request_id: str | None = None
    model: str | None = None
    service: str | None = None
    error_code: str | None = None
    failure_stage: str | None = None
    show_all: bool = False


@dataclass(slots=True)
class _FollowFileState:
    inode: int
    offset: int = 0
    partial: str = ""
    line_number: int = 0


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_env_value(text: str, key: str) -> str:
    value = ""
    for line in text.splitlines():
        match = re.match(rf"^\s*(?:export\s+)?{key}\s*=(.*)$", line)
        if not match:
            continue
        tokens = shlex.split(match.group(1), comments=True)
        if len(tokens) > 1:
            raise LogQueryError(f"Quote values containing spaces: {key}")
        value = tokens[0] if tokens else ""
    return value


def _validate_literal_absolute_path(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise LogQueryError(f"{label} must be an absolute path")
    if any(part == ".." for part in candidate.parts) or any(
        character in value for character in "$\n\r'"
    ):
        raise LogQueryError(f"{label} must be a literal absolute path")
    resolved = candidate.resolve()
    if resolved == Path("/"):
        raise LogQueryError(f"{label} must not be /")
    return resolved


def resolve_logs_root(
    *,
    env_file: str | Path = ".env",
    logs_dir: str | Path | None = None,
) -> Path:
    """Resolve the log root using the same fallback semantics as Compose prep."""
    if logs_dir is not None:
        root = _validate_literal_absolute_path(str(logs_dir), label="--logs-dir")
    else:
        env_path = Path(env_file).expanduser()
        if not env_path.is_absolute():
            env_path = (Path.cwd() / env_path).resolve()
        source = env_path if env_path.exists() else _repository_root() / ".env.template"
        text = source.read_text(encoding="utf-8") if source.exists() else ""
        configured = _read_env_value(text, "LOGS_HOST_PATH")
        if configured:
            root = _validate_literal_absolute_path(
                configured,
                label="LOGS_HOST_PATH",
            )
        else:
            root = (env_path.parent.parent / f"{env_path.parent.name}-logs").resolve()

    if not root.exists() or not root.is_dir():
        raise LogQueryError(f"log root does not exist or is not a directory: {root}")
    return root


def validate_service(service: str | None) -> str | None:
    """Validate the finite service vocabulary before joining any path."""
    if service is None:
        return None
    if service not in SERVICE_NAMES:
        raise LogQueryError(
            f"unsupported service {service!r}; choose one of {', '.join(SERVICE_NAMES)}"
        )
    return service


def _service_directories(root: Path, service: str | None) -> list[tuple[str, Path]]:
    selected = [validate_service(service)] if service else list(SERVICE_NAMES)
    result: list[tuple[str, Path]] = []
    for service_name in selected:
        assert service_name is not None
        service_dir = (root / service_name).resolve()
        try:
            service_dir.relative_to(root)
        except ValueError as exc:
            raise LogQueryError("service path escapes the log root") from exc
        result.append((service_name, service_dir))
    return result


def discover_event_files(root: str | Path, service: str | None = None) -> list[Path]:
    """Discover event files and their rotations directly below service roots."""
    root_path = Path(root).resolve()
    files: list[Path] = []
    for _service_name, service_dir in _service_directories(root_path, service):
        if not service_dir.is_dir():
            continue
        for path in sorted(service_dir.glob("events-*.jsonl*")):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(service_dir)
            except ValueError as exc:
                raise LogQueryError(f"event file escapes service root: {path}") from exc
            files.append(resolved)
    return files


def parse_duration(value: str) -> timedelta:
    """Parse the CLI's bounded relative duration syntax."""
    match = _DURATION_RE.fullmatch(value.strip().lower())
    if match is None:
        raise LogQueryError("duration must look like 30s, 15m, 1h, or 2d")
    amount = float(match.group("value"))
    seconds = amount * {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }[match.group("unit")]
    return timedelta(seconds=seconds)


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp without inventing one for missing records."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _service_for_path(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    if not relative.parts or relative.parts[0] not in SERVICE_NAMES:
        return "unknown"
    return relative.parts[0]


def _record_matches(record: ApplicationRecord, filters: QueryFilters) -> bool:
    fields = record.fields
    if filters.service and record.service != filters.service:
        return False
    if filters.request_id is not None and fields.get("request_id") != filters.request_id:
        return False
    if filters.model is not None and fields.get("model") != filters.model:
        return False
    if filters.error_code is not None and fields.get("error_code") != filters.error_code:
        return False
    if filters.failure_stage is not None and fields.get("failure_stage") != filters.failure_stage:
        return False
    if filters.since is not None:
        timestamp = record.timestamp
        if timestamp is not None and timestamp < filters.since:
            return False
    return True


def is_important_record(record: ApplicationRecord) -> bool:
    """Return whether a record belongs in the default incident summary."""
    fields = record.fields
    level = str(fields.get("level", "")).upper()
    if level in {"WARNING", "ERROR", "CRITICAL"}:
        return True
    event = str(fields.get("event", ""))
    outcome = str(fields.get("outcome", ""))
    if outcome in {"error", "rejected", "timeout", "cancelled"}:
        return True
    if fields.get("slow") is True:
        return True
    return any(marker in event for marker in _IMPORTANT_EVENTS)


def _read_event_file(
    path: Path,
    *,
    root: Path,
    filters: QueryFilters,
    include_unimportant: bool,
    follow: bool = False,
    partial: str = "",
) -> tuple[list[ApplicationRecord], list[str], str]:
    records: list[ApplicationRecord] = []
    diagnostics: list[str] = []
    service = _service_for_path(root, path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            text = partial + handle.read()
    except (OSError, UnicodeError) as error:
        diagnostics.append(f"{path}: unable to read: {type(error).__name__}")
        return records, diagnostics, ""

    complete_lines = text.splitlines(keepends=True)
    remainder = ""
    if complete_lines and not complete_lines[-1].endswith(("\n", "\r")):
        remainder = complete_lines.pop()
    for line_number, line in enumerate(complete_lines, start=1):
        raw_line = line.rstrip("\r\n")
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            diagnostics.append(f"{path}:{line_number}: invalid JSON: {error.msg}")
            continue
        if not isinstance(value, dict):
            diagnostics.append(f"{path}:{line_number}: JSON record is not an object")
            continue
        record = ApplicationRecord(
            fields=value,
            path=path,
            line_number=line_number,
            service=service,
            raw_line=raw_line,
        )
        if _record_matches(record, filters) and (
            include_unimportant or is_important_record(record)
        ):
            records.append(record)
    if remainder and not follow:
        diagnostics.append(f"{path}:{len(complete_lines) + 1}: partial JSON line")
    return records, diagnostics, remainder


def query_application_records(
    root: str | Path,
    *,
    filters: QueryFilters | None = None,
) -> tuple[list[ApplicationRecord], list[str]]:
    """Read finite event files, returning records and visible diagnostics."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise LogQueryError(f"log root does not exist or is not a directory: {root_path}")
    active_filters = filters or QueryFilters()
    include_unimportant = active_filters.show_all or any(
        value is not None
        for value in (
            active_filters.request_id,
            active_filters.model,
            active_filters.error_code,
            active_filters.failure_stage,
        )
    )
    records: list[ApplicationRecord] = []
    diagnostics: list[str] = []
    for path in discover_event_files(root_path, active_filters.service):
        found, issues, _remainder = _read_event_file(
            path,
            root=root_path,
            filters=active_filters,
            include_unimportant=include_unimportant,
        )
        records.extend(found)
        diagnostics.extend(issues)
    records.sort(key=_record_sort_key)
    return records, diagnostics


def _record_sort_key(record: ApplicationRecord) -> tuple[int, datetime, str, int]:
    timestamp = record.timestamp
    if timestamp is None:
        # Unknown-time records stay in file/line order after timestamped records.
        return (1, datetime.max.replace(tzinfo=timezone.utc), str(record.path), record.line_number)
    return (0, timestamp, str(record.path), record.line_number)


def _current_session(ray_dir: Path) -> Path | None:
    latest = ray_dir / "session_latest"
    if latest.is_symlink():
        target = latest.resolve()
        if target.is_dir() and target.parent == ray_dir and target.name.startswith("session_"):
            return target
        return None
    if latest.is_dir() and latest.name.startswith("session_"):
        return latest.resolve()
    sessions = sorted(
        path.resolve()
        for path in ray_dir.glob("session_*")
        if path.is_dir() and not path.is_symlink()
    )
    return sessions[-1] if sessions else None


def discover_infra_files(
    root: str | Path,
    *,
    service: str | None = None,
    all_sessions: bool = False,
) -> list[RawLine]:
    """Read Ray session log files once, excluding the ``session_latest`` alias."""
    root_path = Path(root).resolve()
    lines: list[RawLine] = []
    for service_name, service_dir in _service_directories(root_path, service):
        ray_dir = service_dir / "ray"
        current = _current_session(ray_dir) if ray_dir.is_dir() else None
        sessions = (
            sorted(
                path.resolve()
                for path in ray_dir.glob("session_*")
                if path.is_dir()
                and not path.is_symlink()
                and path.name.startswith("session_")
            )
            if ray_dir.is_dir()
            else []
        )
        selected = sessions if all_sessions else ([current] if current else [])
        seen_sessions: set[Path] = set()
        for session in selected:
            if session is None or session in seen_sessions:
                continue
            seen_sessions.add(session)
            logs_dir = session / "logs"
            if not logs_dir.is_dir():
                continue
            for path in sorted(item for item in logs_dir.rglob("*") if item.is_file() and not item.is_symlink()):
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as handle:
                        for line_number, raw_line in enumerate(handle, start=1):
                            text = raw_line.rstrip("\r\n")
                            lines.append(
                                RawLine(
                                    path=path,
                                    line_number=line_number,
                                    service=service_name,
                                    text=text,
                                    timestamp=_line_timestamp(text),
                                )
                            )
                except OSError as error:
                    lines.append(
                        RawLine(
                            path=path,
                            line_number=0,
                            service=service_name,
                            text=f"[unreadable: {type(error).__name__}]",
                        )
                    )
    return lines


def _line_timestamp(text: str) -> datetime | None:
    match = _TIMESTAMP_RE.search(text)
    return parse_timestamp(match.group("timestamp")) if match else None


def compute_stats(root: str | Path, service: str | None = None) -> LogStats:
    """Count event files, current Ray logs, other session contents, and history."""
    root_path = Path(root).resolve()
    events = FileStats()
    ray_logs = FileStats()
    ray_other = FileStats()
    historical = FileStats()
    for _service_name, service_dir in _service_directories(root_path, service):
        for path in discover_event_files(root_path, _service_name):
            try:
                stat = path.stat()
                modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                events = events.add(stat.st_size, modified_at)
            except OSError:
                continue
        ray_dir = service_dir / "ray"
        if not ray_dir.is_dir():
            continue
        current = _current_session(ray_dir)
        current = current.resolve() if current else None
        sessions = sorted(
            path.resolve()
            for path in ray_dir.glob("session_*")
            if path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("session_")
        )
        for session in sessions:
            category = "current" if current is not None and session == current else "historical"
            for path in session.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    stat = path.stat()
                    size = stat.st_size
                    modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                except OSError:
                    continue
                if category == "historical":
                    historical = historical.add(size, modified_at)
                elif "logs" in path.relative_to(session).parts:
                    ray_logs = ray_logs.add(size, modified_at)
                else:
                    ray_other = ray_other.add(size, modified_at)
    return LogStats(
        events=events,
        ray_logs=ray_logs,
        ray_other=ray_other,
        historical_sessions=historical,
    )


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _record_text(root: Path, record: ApplicationRecord, *, detailed: bool) -> str:
    fields = record.fields
    timestamp = fields.get("timestamp", "-")
    identity = "/".join(
        str(fields.get(key))
        for key in ("physical_service", "process_role")
        if fields.get(key) is not None
    ) or record.service
    summary_keys = (
        "model",
        "request_id",
        "outcome",
        "error_code",
        "failure_stage",
        "timeout_kind",
        "duration_ms",
        "status_code",
    )
    summary = " ".join(
        f"{key}={fields[key]}" for key in summary_keys if key in fields
    )
    location = f"{_display_path(root, record.path)}:{record.line_number}"
    if detailed:
        payload = dict(fields)
        payload["source_path"] = _display_path(root, record.path)
        payload["source_line"] = record.line_number
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{timestamp} {identity} {fields.get('event', 'unknown')} {summary} [{location}]".rstrip()


def render_records(
    root: str | Path,
    records: Sequence[ApplicationRecord],
    *,
    detailed: bool = False,
    json_output: bool = False,
) -> str:
    """Render a stable text-oriented view with source attribution."""
    root_path = Path(root).resolve()
    if not records:
        return "No matching application events.\n"
    rendered = [
        _record_text(root_path, record, detailed=detailed or json_output)
        for record in records
    ]
    return (
        (
            "\n".join(rendered) + "\n"
            if json_output
            else "# Order uses parsed timestamps where available; it is not a distributed causal order.\n"
            + "\n".join(rendered)
            + "\n"
        )
    )


def render_raw(root: str | Path, lines: Iterable[RawLine]) -> str:
    """Preserve raw text while adding only source location."""
    root_path = Path(root).resolve()
    rendered = [
        f"{_display_path(root_path, line.path)}:{line.line_number}\t{line.text}"
        for line in lines
    ]
    return "\n".join(rendered) + ("\n" if rendered else "No raw log lines.\n")


def render_stats(stats: LogStats) -> str:
    """Render storage totals without claiming missing records are healthy."""
    rows = (
        ("events", stats.events),
        ("ray_logs", stats.ray_logs),
        ("ray_other_session_contents", stats.ray_other),
        ("historical_sessions", stats.historical_sessions),
    )

    def format_time(value: datetime | None) -> str:
        if value is None:
            return "-"
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )

    return "\n".join(
        f"{name}: files={value.files} bytes={value.bytes} "
        f"oldest={format_time(value.oldest)} newest={format_time(value.newest)}"
        for name, value in rows
    ) + "\n"


class FollowReader:
    """Incrementally read newly written event files and rotations."""

    def __init__(self, root: Path, filters: QueryFilters, *, include_unimportant: bool) -> None:
        self.root = root.resolve()
        self.filters = filters
        self.include_unimportant = include_unimportant
        self._states: dict[Path, _FollowFileState] = {}
        self._inode_states: dict[int, _FollowFileState] = {}
        self._reported_diagnostics: set[str] = set()

    def poll(self) -> tuple[list[ApplicationRecord], list[str]]:
        records: list[ApplicationRecord] = []
        diagnostics: list[str] = []
        for path in discover_event_files(self.root, self.filters.service):
            try:
                stat = path.stat()
            except OSError as error:
                diagnostics.append(f"{path}: unable to stat: {type(error).__name__}")
                continue
            state = self._states.get(path) or self._inode_states.get(stat.st_ino)
            if (
                state is None
                or state.inode != stat.st_ino
                or stat.st_size < state.offset
            ):
                state = _FollowFileState(inode=stat.st_ino)
            try:
                with path.open("r", encoding="utf-8") as handle:
                    handle.seek(state.offset)
                    chunk = state.partial + handle.read()
                    state.offset = handle.tell()
            except (OSError, UnicodeError) as error:
                message = f"{path}: unable to read: {type(error).__name__}"
                if message not in self._reported_diagnostics:
                    diagnostics.append(message)
                    self._reported_diagnostics.add(message)
                continue
            complete_lines = chunk.splitlines(keepends=True)
            state.partial = ""
            if complete_lines and not complete_lines[-1].endswith(("\n", "\r")):
                state.partial = complete_lines.pop()
            service = _service_for_path(self.root, path)
            start_line = state.line_number + 1
            for line in complete_lines:
                raw_line = line.rstrip("\r\n")
                if not raw_line.strip():
                    start_line += 1
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    message = f"{path}:{start_line}: invalid JSON: {error.msg}"
                    if message not in self._reported_diagnostics:
                        diagnostics.append(message)
                        self._reported_diagnostics.add(message)
                    start_line += 1
                    continue
                if isinstance(value, dict):
                    record = ApplicationRecord(value, path, start_line, service, raw_line)
                    if _record_matches(record, self.filters) and (
                        self.include_unimportant or is_important_record(record)
                    ):
                        records.append(record)
                else:
                    diagnostics.append(f"{path}:{start_line}: JSON record is not an object")
                start_line += 1
            state.line_number += len(complete_lines)
            self._states[path] = state
            self._inode_states[stat.st_ino] = state
        records.sort(key=_record_sort_key)
        return records, diagnostics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", help="explicit absolute host log root")
    parser.add_argument("--env-file", default=".env", help="Compose-style env file")
    parser.add_argument("--since", help="relative window such as 30m or 1h")
    parser.add_argument("--request-id")
    parser.add_argument("--model")
    parser.add_argument("--service", choices=SERVICE_NAMES)
    parser.add_argument("--error-code")
    parser.add_argument("--failure-stage")
    parser.add_argument("--infra", action="store_true", help="include current Ray session text")
    parser.add_argument("--all-sessions", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", "--interval", type=float, default=1.0)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--all", dest="show_all", action="store_true")
    parser.add_argument("--json", dest="json_output", action="store_true")
    return parser


def _filters_from_args(args: argparse.Namespace) -> QueryFilters:
    since = None
    if args.since:
        since = datetime.now(timezone.utc) - parse_duration(args.since)
    return QueryFilters(
        since=since,
        request_id=args.request_id,
        model=args.model,
        service=args.service,
        error_code=args.error_code,
        failure_stage=args.failure_stage,
        show_all=args.show_all,
    )


def _finite_raw_lines(root: Path, filters: QueryFilters, *, infra: bool, all_sessions: bool) -> list[RawLine]:
    lines: list[RawLine] = []
    for path in discover_event_files(root, filters.service):
        service = _service_for_path(root, path)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for number, text in enumerate(handle, start=1):
                    raw = text.rstrip("\r\n")
                    timestamp = None
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        value = None
                    if isinstance(value, dict):
                        record = ApplicationRecord(value, path, number, service, raw)
                        if not _record_matches(record, filters):
                            continue
                        timestamp = record.timestamp
                    elif any(
                        value is not None
                        for value in (
                            filters.request_id,
                            filters.model,
                            filters.error_code,
                            filters.failure_stage,
                        )
                    ):
                        continue
                    lines.append(RawLine(path, number, service, raw, timestamp))
        except OSError as error:
            lines.append(RawLine(path, 0, service, f"[unreadable: {type(error).__name__}]"))
    if infra:
        lines.extend(discover_infra_files(root, service=filters.service, all_sessions=all_sessions))
    if filters.since is not None:
        lines = [line for line in lines if line.timestamp is None or line.timestamp >= filters.since]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local query CLI."""
    args = _build_parser().parse_args(argv)
    if args.poll_interval <= 0:
        _build_parser().error("--poll-interval must be positive")
    if args.stats and (args.follow or args.raw or args.infra or args.request_id or args.model):
        _build_parser().error("--stats cannot be combined with a query mode")
    try:
        root = resolve_logs_root(env_file=args.env_file, logs_dir=args.logs_dir)
        filters = _filters_from_args(args)
        if args.stats:
            sys.stdout.write(render_stats(compute_stats(root, args.service)))
            return 0
        if args.follow:
            if args.raw:
                raise LogQueryError("--raw cannot be combined with --follow")
            reader = FollowReader(
                root,
                filters,
                include_unimportant=args.show_all
                or any(
                    value is not None
                    for value in (
                        args.request_id,
                        args.model,
                        args.error_code,
                        args.failure_stage,
                    )
                ),
            )
            try:
                while True:
                    records, diagnostics = reader.poll()
                    if records:
                        sys.stdout.write(
                            render_records(
                                root,
                                records,
                                detailed=args.request_id is not None,
                                json_output=args.json_output,
                            )
                        )
                        sys.stdout.flush()
                    for diagnostic in diagnostics:
                        print(f"diagnostic: {diagnostic}", file=sys.stderr, flush=True)
                    time.sleep(max(0.1, args.poll_interval))
            except KeyboardInterrupt:
                return 0

        if args.raw:
            raw_lines = _finite_raw_lines(
                root,
                filters,
                infra=args.infra,
                all_sessions=args.all_sessions,
            )
            sys.stdout.write(
                render_raw(root, raw_lines)
            )
            return 1 if any(line.line_number == 0 for line in raw_lines) else 0

        records, diagnostics = query_application_records(root, filters=filters)
        sys.stdout.write(
            render_records(
                root,
                records,
                detailed=args.request_id is not None,
                json_output=args.json_output,
            )
        )
        infra_unreadable = False
        if args.infra:
            infra_lines = discover_infra_files(
                root,
                service=filters.service,
                all_sessions=args.all_sessions,
            )
            infra_unreadable = any(line.line_number == 0 for line in infra_lines)
            if infra_lines:
                print("# Infrastructure source lines")
                sys.stdout.write(render_raw(root, infra_lines))
        for diagnostic in diagnostics:
            print(f"diagnostic: {diagnostic}", file=sys.stderr)
        return 1 if diagnostics or infra_unreadable else 0
    except (LogQueryError, OSError, UnicodeError) as error:
        print(f"logs.py: {error}", file=sys.stderr)
        return 2


__all__ = [
    "ApplicationRecord",
    "FileStats",
    "FollowReader",
    "LogQueryError",
    "LogStats",
    "QueryFilters",
    "RawLine",
    "SERVICE_NAMES",
    "compute_stats",
    "discover_event_files",
    "discover_infra_files",
    "is_important_record",
    "main",
    "parse_duration",
    "parse_timestamp",
    "query_application_records",
    "render_records",
    "render_raw",
    "render_stats",
    "resolve_logs_root",
]
