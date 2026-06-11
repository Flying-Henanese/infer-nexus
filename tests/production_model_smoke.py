#!/usr/bin/env python3
"""生产环境模型接口冒烟测试脚本。

这个脚本只依赖 Python 标准库，方便直接复制到生产节点运行，不需要额外安装
pytest、httpx 等测试依赖。

常用方式如下，默认都在仓库根目录执行：

    # 只查看本次会测试哪些非 MinerU 模型，不发送任何 HTTP 请求。
    python tests/production_model_smoke.py --list-only

    # 对生产服务执行完整冒烟测试；单个能力失败后继续测试其它能力，
    # 最后统一输出通过/失败汇总。
    python tests/production_model_smoke.py --base-url http://192.168.0.194:8000 --continue-on-error

    # 推荐作为生产准入检查使用：额外要求 /v1/models 必须暴露
    # config/models.yaml 中所有非 MinerU 的模型 alias。
    python tests/production_model_smoke.py --base-url http://192.168.0.194:8000 --strict-model-list --continue-on-error

    # 只测试某一个模型或某一类模型。
    python tests/production_model_smoke.py --base-url http://192.168.0.194:8000 --only qwen3.5-9b --continue-on-error

    # 调大 qwen3.5-9b 同前缀缓存亲和性测试的重复次数。
    python tests/production_model_smoke.py --base-url http://192.168.0.194:8000 --only qwen3.5-9b --cache-affinity-repeats 10 --continue-on-error

    # 如果服务通过自定义 header 或响应字段暴露 replica/instance 标识，
    # 可以显式指定给 qwen3.5-9b 的前缀缓存亲和性测试使用。
    # 参数值可以是响应 header 名，也可以是点号分隔的 JSON 路径。
    python tests/production_model_smoke.py --base-url http://192.168.0.194:8000 --only qwen3.5-9b --cache-affinity-marker X-Ray-Serve-Replica-ID --continue-on-error

使用前请让部署同事把下面的 MULTIMODAL_IMAGE_PATH 常量改成生产节点上的真实图片路径。
如果该常量为空，多模态图片测试会跳过，其它测试仍会继续。

脚本默认读取 config/models.yaml，排除 alias/name 包含 "mineru" 的模型。
当前覆盖项包括：/healthz、/v1/models、非流式 chat、流式 chat 和
data: [DONE] 校验、工具调用、多模态图片输入、embedding、批量 embedding、
rerank 排序，以及 qwen3.5-9b 的前缀缓存亲和性。
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request


# 部署同事填写生产节点上的真实测试图片路径，例如：
# MULTIMODAL_IMAGE_PATH = "/data/test_images/receipt.jpg"
# 留空时会跳过 qwen3.5-9b 的多模态图片测试。
MULTIMODAL_IMAGE_PATH = ""

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


@dataclass
class ModelSpec:
    name: str
    alias: str
    task: str
    capabilities: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @property
    def is_vision(self) -> bool:
        values = {value.lower() for value in self.capabilities + self.labels}
        return "vision" in values or "multimodal" in values


@dataclass
class CheckResult:
    name: str
    ok: bool
    elapsed_ms: float
    detail: str


@dataclass
class JsonHttpResponse:
    elapsed_ms: float
    data: dict[str, Any]
    headers: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run production smoke tests for all configured models except excluded aliases."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("INFER_NEXUS_BASE_URL", "http://127.0.0.1:8000"),
        help="Infer-nexus base URL. Defaults to INFER_NEXUS_BASE_URL or http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "models.yaml"),
        help="Path to config/models.yaml.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["mineru"],
        help="Model alias/name substring to exclude. Can be passed multiple times. Default: mineru.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only test aliases/names containing this substring. Can be passed multiple times.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--chat-tokens", type=int, default=64, help="Max tokens for chat smoke checks.")
    parser.add_argument(
        "--cache-affinity-repeats",
        type=int,
        default=6,
        help="Repeated same-prefix requests for the qwen3.5-9b cache-affinity check.",
    )
    parser.add_argument(
        "--cache-affinity-marker",
        action="append",
        default=[],
        help=(
            "Header name or dotted JSON path that identifies the serving replica/instance. "
            "If omitted, the script auto-detects common names."
        ),
    )
    parser.add_argument(
        "--strict-model-list",
        action="store_true",
        help="Fail if /v1/models does not contain every non-excluded configured alias.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining checks after failures and return non-zero at the end.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print selected models and exit without sending HTTP requests.",
    )
    return parser


def strip_inline_comment(value: str) -> str:
    in_quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            in_quote = None if in_quote == char else char
        elif char == "#" and in_quote is None:
            return value[:index].rstrip()
    return value.strip()


def scalar(value: str) -> str:
    value = strip_inline_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_models_yaml(path: Path) -> list[ModelSpec]:
    """Parse the small subset of YAML used by config/models.yaml."""
    models: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.startswith("#"):
            continue

        if indent == 2 and line.startswith("- "):
            if current:
                models.append(current)
            current = {}
            current_list_key = None
            rest = line[2:].strip()
            if rest:
                key, _, value = rest.partition(":")
                current[key.strip()] = scalar(value)
            continue

        if current is None:
            continue

        if indent == 4 and line.endswith(":"):
            key = line[:-1].strip()
            if key in {"capabilities", "labels"}:
                current[key] = []
                current_list_key = key
            else:
                current_list_key = None
            continue

        if indent == 4 and ":" in line:
            key, _, value = line.partition(":")
            current[key.strip()] = scalar(value)
            current_list_key = None
            continue

        if indent == 6 and line.startswith("- ") and current_list_key:
            current[current_list_key].append(scalar(line[2:]))

    if current:
        models.append(current)

    specs: list[ModelSpec] = []
    for item in models:
        alias = str(item.get("alias") or item.get("name") or "").strip()
        task = str(item.get("task") or "").strip()
        name = str(item.get("name") or alias).strip()
        if alias and task:
            specs.append(
                ModelSpec(
                    name=name,
                    alias=alias,
                    task=task,
                    capabilities=list(item.get("capabilities") or []),
                    labels=list(item.get("labels") or []),
                )
            )
    return specs


def now_ms() -> float:
    return time.perf_counter() * 1000


def http_json(method: str, url: str, timeout: float, payload: dict[str, Any] | None = None) -> tuple[float, dict[str, Any]]:
    response = http_json_with_headers(method, url, timeout, payload)
    return response.elapsed_ms, response.data


def http_json_with_headers(
    method: str,
    url: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> JsonHttpResponse:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=body, headers=headers, method=method)
    started = now_ms()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            response_headers = {key.lower(): value for key, value in resp.headers.items()}
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {raw[:1000]}") from exc
    elapsed_ms = now_ms() - started
    return JsonHttpResponse(
        elapsed_ms=elapsed_ms,
        data=json.loads(raw.decode("utf-8")),
        headers=response_headers,
    )


def value_at_json_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def extract_instance_marker(
    response: JsonHttpResponse,
    *,
    markers: list[str],
) -> tuple[str, str] | None:
    candidates = markers or [
        "x-infer-nexus-replica-id",
        "x-infer-nexus-instance-id",
        "x-ray-serve-replica-id",
        "x-serve-replica-id",
        "x-ray-replica-id",
        "x-replica-id",
        "x-instance-id",
        "serve-replica-id",
        "replica_id",
        "replica",
        "instance_id",
        "instance",
        "serve_replica_id",
        "metadata.replica_id",
        "metadata.instance_id",
        "_metadata.replica_id",
        "_metadata.instance_id",
    ]

    for marker in candidates:
        key = marker.lower()
        header_value = response.headers.get(key)
        if header_value:
            return marker, header_value

        body_value = value_at_json_path(response.data, marker)
        if isinstance(body_value, (str, int, float)) and str(body_value):
            return marker, str(body_value)
    return None


def build_image_data_url(image_path: str) -> str | None:
    if not image_path.strip():
        return None
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise ValueError(f"configured MULTIMODAL_IMAGE_PATH does not exist or is not a file: {path}")
    mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(
            f"unsupported image suffix {path.suffix!r}; supported suffixes: {sorted(IMAGE_MIME_TYPES)}"
        )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def http_text(method: str, url: str, timeout: float, payload: dict[str, Any] | None = None) -> tuple[float, str, int]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "*/*"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=body, headers=headers, method=method)
    started = now_ms()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    elapsed_ms = now_ms() - started
    return elapsed_ms, raw.decode("utf-8", errors="replace"), status


def post_sse(url: str, timeout: float, payload: dict[str, Any]) -> tuple[float, list[dict[str, Any]], bool, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[dict[str, Any]] = []
    saw_done = False
    text_parts: list[str] = []
    started = now_ms()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                parsed = json.loads(data)
                chunks.append(parsed)
                for choice in parsed.get("choices", []):
                    delta = choice.get("delta") or {}
                    for key in ("content", "reasoning_content"):
                        content = delta.get(key)
                        if isinstance(content, str):
                            text_parts.append(content)
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {raw[:1000]}") from exc
    return now_ms() - started, chunks, saw_done, "".join(text_parts)


def chat_payload(model: str, prompt: str, max_tokens: int, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer briefly. Do not include hidden reasoning."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def assert_chat_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("chat choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"chat content is empty or invalid: {content!r}")
    return content.strip()


def assert_chat_or_tool_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("chat choice has no message")

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        names: list[str] = []
        for item in tool_calls:
            function = item.get("function") if isinstance(item, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str):
                names.append(name)
        return f"tool_calls={len(tool_calls)}, names={names}"

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return f"text_fallback_chars={len(content.strip())}"
    raise ValueError("tool check produced neither tool_calls nor text content")


def assert_embedding_response(data: dict[str, Any], expected_count: int) -> tuple[int, int]:
    items = data.get("data")
    if not isinstance(items, list) or len(items) != expected_count:
        raise ValueError(f"expected {expected_count} embeddings, got {len(items) if isinstance(items, list) else 'invalid'}")

    dims: set[int] = set()
    for index, item in enumerate(items):
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"embedding item {index} is empty or not a list")
        dims.add(len(embedding))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
            raise ValueError(f"embedding item {index} contains non-finite values")
        if sum(abs(float(value)) for value in embedding) == 0:
            raise ValueError(f"embedding item {index} is all zeros")

    if len(dims) != 1:
        raise ValueError(f"embedding dimensions differ: {sorted(dims)}")

    usage = data.get("usage")
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else -1
    return dims.pop(), int(total_tokens) if isinstance(total_tokens, int) else -1


def assert_rerank_response(data: dict[str, Any], expected_top_index: int) -> tuple[int, float]:
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("rerank response has no results")
    first = results[0]
    if not isinstance(first, dict):
        raise ValueError("rerank result item is not an object")
    top_index = first.get("index")
    score = first.get("relevance_score")
    if not isinstance(top_index, int):
        raise ValueError(f"rerank top index is invalid: {top_index!r}")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"rerank top score is invalid: {score!r}")
    if top_index != expected_top_index:
        raise ValueError(f"expected rerank top index {expected_top_index}, got {top_index}")
    return top_index, float(score)


def run_check(name: str, func: Any, continue_on_error: bool, results: list[CheckResult]) -> None:
    started = now_ms()
    try:
        detail = func()
    except Exception as exc:  # noqa: BLE001 - this is a smoke-test runner.
        elapsed = now_ms() - started
        results.append(CheckResult(name=name, ok=False, elapsed_ms=elapsed, detail=str(exc)))
        print(f"[FAIL] {name} ({elapsed:.1f} ms) {exc}")
        if not continue_on_error:
            raise
    else:
        elapsed = now_ms() - started
        results.append(CheckResult(name=name, ok=True, elapsed_ms=elapsed, detail=str(detail)))
        print(f"[ OK ] {name} ({elapsed:.1f} ms) {detail}")


def model_matches(value: str, patterns: list[str]) -> bool:
    folded = value.lower()
    return any(pattern.lower() in folded for pattern in patterns)


def remote_model_ids(base_url: str, timeout: float) -> tuple[float, set[str], dict[str, Any]]:
    elapsed, data = http_json("GET", f"{base_url}/v1/models", timeout)
    ids: set[str] = set()
    for item in data.get("data", []):
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str):
                ids.add(model_id)
    return elapsed, ids, data


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")
    specs = parse_models_yaml(Path(args.config))
    excluded = args.exclude or []
    if args.only:
        specs = [spec for spec in specs if model_matches(spec.alias, args.only) or model_matches(spec.name, args.only)]
    specs = [
        spec
        for spec in specs
        if not model_matches(spec.alias, excluded) and not model_matches(spec.name, excluded)
    ]

    if not specs:
        print("No configured models selected for testing.", file=sys.stderr)
        return 2

    print(f"Base URL: {base_url}")
    print("Selected models:")
    for spec in specs:
        flags = ", ".join([spec.task] + spec.capabilities + spec.labels)
        print(f"  - {spec.alias} ({flags})")
    if args.list_only:
        return 0

    results: list[CheckResult] = []
    remote_ids: set[str] = set()

    run_check(
        "healthz",
        lambda: (
            lambda response: f"status={response[2]}, body={response[1][:120]!r}"
        )(http_text("GET", f"{base_url}/healthz", args.timeout)),
        args.continue_on_error,
        results,
    )

    def check_models() -> str:
        nonlocal remote_ids
        elapsed, ids, _data = remote_model_ids(base_url, args.timeout)
        remote_ids = ids
        missing = [spec.alias for spec in specs if spec.alias not in ids and spec.name not in ids]
        if missing and args.strict_model_list:
            raise ValueError(f"configured aliases missing from /v1/models: {missing}")
        missing_text = f", missing={missing}" if missing else ""
        return f"remote_count={len(ids)}, elapsed={elapsed:.1f} ms{missing_text}"

    run_check("models.list", check_models, args.continue_on_error, results)

    chat_models = [spec for spec in specs if spec.task == "chat"]
    embedding_models = [spec for spec in specs if spec.task == "embedding"]
    rerank_models = [spec for spec in specs if spec.task == "rerank"]

    for spec in chat_models:
        chat_url = f"{base_url}/v1/chat/completions"

        def check_chat_non_stream(spec: ModelSpec = spec) -> str:
            payload = chat_payload(spec.alias, "Return exactly this phrase: infer-nexus-ok", args.chat_tokens)
            elapsed, data = http_json("POST", chat_url, args.timeout, payload)
            content = assert_chat_response(data)
            return f"chars={len(content)}, elapsed={elapsed:.1f} ms, sample={content[:80]!r}"

        run_check(f"chat.non_stream.{spec.alias}", check_chat_non_stream, args.continue_on_error, results)

        def check_chat_stream(spec: ModelSpec = spec) -> str:
            payload = chat_payload(spec.alias, "Say: stream-ok", args.chat_tokens, stream=True)
            elapsed, chunks, saw_done, text = post_sse(chat_url, args.timeout, payload)
            if not chunks:
                raise ValueError("stream returned no JSON chunks")
            if not saw_done:
                raise ValueError("stream did not end with data: [DONE]")
            if not text.strip():
                raise ValueError("stream returned no text delta content")
            return f"chunks={len(chunks)}, done={saw_done}, chars={len(text)}, elapsed={elapsed:.1f} ms"

        run_check(f"chat.stream.{spec.alias}", check_chat_stream, args.continue_on_error, results)

        def check_chat_tools(spec: ModelSpec = spec) -> str:
            payload = chat_payload(
                spec.alias,
                "Use the available tool to report the weather for Shanghai. Do not guess.",
                args.chat_tokens,
            )
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
            payload["tool_choice"] = "auto"
            elapsed, data = http_json("POST", chat_url, args.timeout, payload)
            detail = assert_chat_or_tool_response(data)
            return f"{detail}, elapsed={elapsed:.1f} ms"

        run_check(f"chat.tools.{spec.alias}", check_chat_tools, args.continue_on_error, results)

        if spec.is_vision:

            def check_vision(spec: ModelSpec = spec) -> str:
                image_data_url = build_image_data_url(MULTIMODAL_IMAGE_PATH)
                if image_data_url is None:
                    return "skipped: MULTIMODAL_IMAGE_PATH is empty"
                payload = {
                    "model": spec.alias,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is visible in this tiny image? Answer in one short sentence."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url,
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0,
                    "max_completion_tokens": args.chat_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                elapsed, data = http_json("POST", chat_url, args.timeout, payload)
                content = assert_chat_response(data)
                return f"chars={len(content)}, elapsed={elapsed:.1f} ms, sample={content[:80]!r}"

            run_check(f"chat.vision.{spec.alias}", check_vision, args.continue_on_error, results)

        if spec.alias == "qwen3.5-9b":

            def check_prefix_cache_affinity(spec: ModelSpec = spec) -> str:
                if args.cache_affinity_repeats < 2:
                    raise ValueError("--cache-affinity-repeats must be at least 2")

                shared_prefix = (
                    "You are running a production prefix-cache affinity test. "
                    "Keep this exact prefix stable across requests. "
                    "The router should send requests sharing this prefix to the same replica when possible. "
                    "Context marker: qwen35-prefix-cache-affinity-stable-prefix-v1. "
                )
                markers: list[tuple[str, str]] = []
                latencies: list[float] = []
                samples: list[str] = []

                for index in range(args.cache_affinity_repeats):
                    payload = {
                        "model": spec.alias,
                        "messages": [
                            {"role": "system", "content": shared_prefix},
                            {
                                "role": "user",
                                "content": (
                                    "Return only the token AFFINITY-OK and the request number "
                                    f"{index}."
                                ),
                            },
                        ],
                        "temperature": 0,
                        "max_completion_tokens": min(args.chat_tokens, 32),
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    response = http_json_with_headers("POST", chat_url, args.timeout, payload)
                    content = assert_chat_response(response.data)
                    marker = extract_instance_marker(response, markers=args.cache_affinity_marker)
                    if marker is None:
                        available_headers = sorted(response.headers)
                        raise ValueError(
                            "cannot accurately verify prefix-cache affinity because the response "
                            "does not expose a replica/instance marker. Pass --cache-affinity-marker "
                            "with a header name or JSON path, or expose a header such as "
                            "X-Infer-Nexus-Replica-ID from the serving replica. "
                            f"available_headers={available_headers}"
                        )
                    markers.append(marker)
                    latencies.append(response.elapsed_ms)
                    samples.append(content[:40])

                marker_names = {name for name, _value in markers}
                marker_values = [value for _name, value in markers]
                unique_values = sorted(set(marker_values))
                if len(marker_names) != 1:
                    raise ValueError(f"instance marker source changed across requests: {markers}")
                if len(unique_values) != 1:
                    raise ValueError(
                        "same-prefix requests did not consistently hit the same instance: "
                        f"markers={marker_values}, latencies_ms={[round(value, 1) for value in latencies]}"
                    )
                avg_ms = sum(latencies) / len(latencies)
                return (
                    f"marker={markers[0][0]}:{unique_values[0]}, repeats={len(markers)}, "
                    f"avg={avg_ms:.1f} ms, max={max(latencies):.1f} ms, samples={samples[:2]}"
                )

            run_check(
                f"chat.prefix_cache_affinity.{spec.alias}",
                check_prefix_cache_affinity,
                args.continue_on_error,
                results,
            )

    for spec in embedding_models:
        embedding_url = f"{base_url}/v1/embeddings"

        def check_embedding_single(spec: ModelSpec = spec) -> str:
            payload = {"model": spec.alias, "input": "infer nexus embedding smoke test"}
            elapsed, data = http_json("POST", embedding_url, args.timeout, payload)
            dim, total_tokens = assert_embedding_response(data, 1)
            return f"count=1, dim={dim}, tokens={total_tokens}, elapsed={elapsed:.1f} ms"

        run_check(f"embedding.single.{spec.alias}", check_embedding_single, args.continue_on_error, results)

        def check_embedding_batch(spec: ModelSpec = spec) -> str:
            payload = {
                "model": spec.alias,
                "input": [
                    "Paris is the capital of France.",
                    "Python is a programming language.",
                    "Text embeddings should keep a stable dimensionality.",
                ],
            }
            elapsed, data = http_json("POST", embedding_url, args.timeout, payload)
            dim, total_tokens = assert_embedding_response(data, 3)
            return f"count=3, dim={dim}, tokens={total_tokens}, elapsed={elapsed:.1f} ms"

        run_check(f"embedding.batch.{spec.alias}", check_embedding_batch, args.continue_on_error, results)

    for spec in rerank_models:
        rerank_url = f"{base_url}/v1/rerank"

        def check_rerank(spec: ModelSpec = spec) -> str:
            payload = {
                "model": spec.alias,
                "query": "Which document explains Python programming?",
                "documents": [
                    "Paris is the capital city of France.",
                    "Python is a programming language used for software and data work.",
                    "The Pacific Ocean is the largest ocean on Earth.",
                ],
                "top_n": 3,
            }
            elapsed, data = http_json("POST", rerank_url, args.timeout, payload)
            top_index, top_score = assert_rerank_response(data, expected_top_index=1)
            return f"top_index={top_index}, top_score={top_score:.6f}, elapsed={elapsed:.1f} ms"

        run_check(f"rerank.ordering.{spec.alias}", check_rerank, args.continue_on_error, results)

    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    print(f"\nSummary: passed={passed}, failed={failed}, total={len(results)}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
