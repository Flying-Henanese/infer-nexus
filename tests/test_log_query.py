"""Deterministic tests for the local application and Ray log query layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infer_nexus.observability.log_query import (
    FollowReader,
    LogQueryError,
    QueryFilters,
    compute_stats,
    discover_infra_files,
    main,
    query_application_records,
    resolve_logs_root,
)


def _event(
    *,
    timestamp: str,
    event: str,
    request_id: str,
    **fields: object,
) -> str:
    payload = {
        "schema_version": 1,
        "timestamp": timestamp,
        "level": "INFO",
        "event": event,
        "physical_service": fields.pop("physical_service", "ray-head"),
        "process_role": fields.pop("process_role", "gateway"),
        "process_instance": fields.pop("process_instance", "instance-a"),
        "request_id": request_id,
        **fields,
    }
    return json.dumps(payload, separators=(",", ":"))


def _write_fixture(root: Path) -> None:
    for service in ("ray-head", "ray-worker", "serve-deployer"):
        (root / service / "ray").mkdir(parents=True)
    (root / "ray-head" / "events-gateway-head.jsonl").write_text(
        "\n".join(
            (
                _event(
                    timestamp="2026-09-20T00:00:01Z",
                    event="request.completed",
                    request_id="req-shared",
                    outcome="success",
                    model="qwen3.5-9b",
                    physical_service="ray-head",
                ),
                _event(
                    timestamp="2026-09-20T00:00:02Z",
                    event="request.failed",
                    request_id="req-failed",
                    outcome="error",
                    error_code="upstream_timeout",
                    failure_stage="serve_handle",
                    timeout_kind="serve_handle",
                    model="qwen3.5-9b",
                    physical_service="ray-head",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "ray-worker" / "events-model-replica-worker.jsonl").write_text(
        _event(
            timestamp="2026-09-20T00:00:03Z",
            event="model.replica.ready",
            request_id="req-shared",
            model="qwen3.5-9b",
            physical_service="ray-worker",
            process_role="model_replica",
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "serve-deployer" / "events-gateway-deployer.jsonl").write_text(
        _event(
            timestamp="2026-09-20T00:00:04Z",
            event="app.ready",
            request_id="req-other",
            physical_service="serve-deployer",
        )
        + "\n",
        encoding="utf-8",
    )

    current = root / "ray-head" / "ray" / "session_20260920_000000"
    (current / "logs").mkdir(parents=True)
    (current / "logs" / "raylet.out").write_text("raylet current\n", encoding="utf-8")
    (current / "runtime.state").write_text("state\n", encoding="utf-8")
    (root / "ray-head" / "ray" / "session_latest").symlink_to(current.name)
    historical = root / "ray-head" / "ray" / "session_20260919_000000"
    (historical / "logs").mkdir(parents=True)
    (historical / "logs" / "raylet.out").write_text("raylet historical\n", encoding="utf-8")


def test_request_query_merges_services_and_preserves_source_attribution(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    records, diagnostics = query_application_records(
        tmp_path,
        filters=QueryFilters(request_id="req-shared"),
    )

    assert diagnostics == []
    assert [(record.service, record.fields["event"]) for record in records] == [
        ("ray-head", "request.completed"),
        ("ray-worker", "model.replica.ready"),
    ]
    assert all(record.fields["request_id"] == "req-shared" for record in records)


def test_default_query_shows_incident_events_and_all_shows_success_events(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    default_records, _ = query_application_records(tmp_path)
    all_records, _ = query_application_records(tmp_path, filters=QueryFilters(show_all=True))

    assert {record.fields["event"] for record in default_records} == {
        "request.failed",
        "model.replica.ready",
        "app.ready",
    }
    assert {record.fields["event"] for record in all_records} == {
        "request.completed",
        "request.failed",
        "model.replica.ready",
        "app.ready",
    }


def test_infra_current_session_excludes_latest_alias_and_stats_separate_categories(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    lines = discover_infra_files(tmp_path, service="ray-head")
    assert [line.text for line in lines] == ["raylet current"]

    stats = compute_stats(tmp_path, service="ray-head")
    assert stats.events.files == 1
    assert stats.ray_logs.files == 1
    assert stats.ray_other.files == 1
    assert stats.historical_sessions.files == 1

    historical = discover_infra_files(tmp_path, service="ray-head", all_sessions=True)
    assert [line.text for line in historical] == ["raylet historical", "raylet current"]


def test_follow_retries_partial_lines_and_discovers_rotation(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    path = tmp_path / "ray-head" / "events-gateway-follow.jsonl"
    path.write_text(
        _event(
            timestamp="2026-09-20T00:01:00Z",
            event="request.completed",
            request_id="req-follow",
            outcome="success",
        )
        + "\n",
        encoding="utf-8",
    )
    reader = FollowReader(tmp_path, QueryFilters(show_all=True), include_unimportant=True)

    first, diagnostics = reader.poll()
    assert diagnostics == []
    assert [record.line_number for record in first if record.fields["request_id"] == "req-follow"] == [1]

    second_line = _event(
        timestamp="2026-09-20T00:01:01Z",
        event="request.failed",
        request_id="req-follow",
        outcome="error",
        error_code="stream_failed",
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(second_line)
    partial, _ = reader.poll()
    assert not [record for record in partial if record.fields["request_id"] == "req-follow"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    complete, _ = reader.poll()
    assert [record.line_number for record in complete if record.fields["request_id"] == "req-follow"] == [2]

    rotated = path.with_name(path.name + ".1")
    path.rename(rotated)
    path.write_text(
        _event(
            timestamp="2026-09-20T00:01:02Z",
            event="request.failed",
            request_id="req-new-file",
            outcome="error",
            error_code="model_backend",
        )
        + "\n",
        encoding="utf-8",
    )
    rotated_records, _ = reader.poll()
    assert [record.fields["request_id"] for record in rotated_records] == ["req-new-file"]


def test_malformed_finite_query_reports_diagnostic_and_nonzero_cli_status(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    malformed = tmp_path / "ray-head" / "events-gateway-bad.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")

    records, diagnostics = query_application_records(tmp_path, filters=QueryFilters(show_all=True))
    assert records
    assert any("invalid JSON" in diagnostic for diagnostic in diagnostics)
    assert main(["--logs-dir", str(tmp_path), "--all"]) == 1


def test_query_path_validation_rejects_root_and_traversal(tmp_path: Path) -> None:
    with pytest.raises(LogQueryError):
        resolve_logs_root(logs_dir="/")
    with pytest.raises(LogQueryError):
        resolve_logs_root(logs_dir=str(tmp_path / ".." / "logs"))
