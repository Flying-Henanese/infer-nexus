#!/usr/bin/env python3
"""Sequential multi-model gateway smoke/load script for local execution."""

from __future__ import annotations

import argparse
import json
import time
from urllib import request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sequential multi-model gateway checks against a local infer-nexus service."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def post_json(url: str, payload: dict[str, object], timeout: float) -> tuple[int, float, dict[str, object]]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        status = resp.status
    elapsed_ms = (time.perf_counter() - started) * 1000
    return status, elapsed_ms, json.loads(raw)


def main() -> int:
    args = build_parser().parse_args()
    base = args.base_url.rstrip("/")

    cases = [
        (
            "qwen3-32b",
            f"{base}/v1/chat/completions",
            {
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": "请只回复OK"}],
                "temperature": 0,
                "max_tokens": 32,
            },
            "chat",
        ),
        (
            "qwen3-vl-chat-8b-instruct",
            f"{base}/v1/chat/completions",
            {
                "model": "qwen3-vl-chat-8b-instruct",
                "messages": [{"role": "user", "content": "请只回复OK"}],
                "temperature": 0,
                "max_tokens": 32,
            },
            "chat",
        ),
        (
            "qwen3-embedding-8b",
            f"{base}/v1/embeddings",
            {"model": "qwen3-embedding-8b", "input": "hello infer nexus"},
            "embedding",
        ),
        (
            "bge-reranker",
            f"{base}/v1/rerank",
            {
                "model": "bge-reranker",
                "query": "what is the capital of france",
                "documents": [
                    "Paris is the capital of France.",
                    "Python is a programming language.",
                    "Berlin is the capital of Germany.",
                ],
                "top_n": 3,
            },
            "rerank",
        ),
    ]

    successes = 0
    failures = 0
    latencies_ms: list[float] = []
    started_all = time.perf_counter()

    for round_index in range(1, args.rounds + 1):
        print(f"===== round {round_index} =====")
        for model, url, payload, kind in cases:
            try:
                status, elapsed_ms, data = post_json(url, payload, args.timeout)
                latencies_ms.append(elapsed_ms)
                successes += 1
                if kind == "chat":
                    content = data["choices"][0]["message"]["content"]
                    print(
                        f"{model} status={status} {elapsed_ms:.2f} ms "
                        f"content={content!r}"
                    )
                elif kind == "embedding":
                    dim = len(data["data"][0]["embedding"])
                    print(
                        f"{model} status={status} {elapsed_ms:.2f} ms "
                        f"dim={dim} tokens={data['usage']['total_tokens']}"
                    )
                else:
                    top = data["results"][0]
                    print(
                        f"{model} status={status} {elapsed_ms:.2f} ms "
                        f"top_index={top['index']} score={top['relevance_score']:.6f}"
                    )
            except Exception as exc:
                failures += 1
                print(f"{model} FAILED error={exc}")

    wall_s = time.perf_counter() - started_all
    print("--- summary ---")
    print(
        json.dumps(
            {
                "rounds": args.rounds,
                "requests": successes + failures,
                "successes": successes,
                "failures": failures,
                "wall_s": round(wall_s, 2),
                "min_ms": round(min(latencies_ms), 2) if latencies_ms else None,
                "max_ms": round(max(latencies_ms), 2) if latencies_ms else None,
                "avg_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else None,
            },
            ensure_ascii=False,
        )
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
