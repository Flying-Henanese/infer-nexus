#!/usr/bin/env python3
"""Sequential local smoke test for embedding models."""

from __future__ import annotations

import argparse
import json
import math
import time
from urllib import request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sequential local embedding smoke tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-embedding-8b")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def validate_embedding_response(data: dict[str, object], expected_count: int) -> tuple[int, int]:
    items = data.get("data")
    if not isinstance(items, list):
        raise ValueError("response field 'data' is not a list")
    if len(items) != expected_count:
        raise ValueError(f"expected {expected_count} embeddings, got {len(items)}")

    dims: set[int] = set()
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"embedding item {item_index} is not an object")

        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"embedding item {item_index} has empty or invalid vector")

        dims.add(len(embedding))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
            raise ValueError(f"embedding item {item_index} contains non-finite values")

        magnitude = math.sqrt(sum(float(value) * float(value) for value in embedding))
        if magnitude == 0:
            raise ValueError(f"embedding item {item_index} is an all-zero vector")

    if len(dims) != 1:
        raise ValueError(f"embedding dimensions are inconsistent: {sorted(dims)}")

    usage = data.get("usage")
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return dims.pop(), int(total_tokens) if isinstance(total_tokens, int) else -1


def post_json(url: str, payload: dict[str, object], timeout: float) -> tuple[float, dict[str, object]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, json.loads(raw)


def main() -> int:
    args = build_parser().parse_args()
    url = f"{args.base_url.rstrip('/')}/v1/embeddings"

    payloads: list[dict[str, object]] = [
        {
            "model": args.model,
            "input": "hello infer nexus",
        },
        {
            "model": args.model,
            "input": "Qwen3 embedding service smoke test.",
        },
        {
            "model": args.model,
            "input": [
                "Paris is the capital of France.",
                "Python is a programming language.",
                "Embeddings map text into dense vectors.",
            ],
        },
        {
            "model": args.model,
            "input": [
                "检索增强生成需要稳定的文本向量。",
                "向量维度应该在同一个模型内保持一致。",
            ],
        },
    ]

    expected_dim: int | None = None
    for index, payload in enumerate(payloads, 1):
        input_value = payload["input"]
        expected_count = len(input_value) if isinstance(input_value, list) else 1

        elapsed_ms, data = post_json(url, payload, args.timeout)
        dim, total_tokens = validate_embedding_response(data, expected_count)
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            raise ValueError(f"dimension changed from {expected_dim} to {dim}")

        print(
            f"#{index} {elapsed_ms:.2f} ms "
            f"count={expected_count} dim={dim} tokens={total_tokens}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
