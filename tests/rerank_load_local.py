#!/usr/bin/env python3
"""Concurrent local load test for rerank models."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
from urllib import request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run concurrent local rerank load tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="bge-reranker")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    url = f"{args.base_url.rstrip('/')}/v1/rerank"
    headers = {"Content-Type": "application/json"}

    cases = [
        (
            "what is the capital of france",
            [
                "Paris is the capital of France.",
                "Python is a programming language.",
                "Berlin is the capital of Germany.",
            ],
        ),
        (
            "who wrote pride and prejudice",
            [
                "Jane Austen wrote Pride and Prejudice.",
                "Pride and Prejudice is a novel.",
                "Shakespeare wrote Hamlet.",
            ],
        ),
        (
            "what planet is known as the red planet",
            [
                "Mars is known as the Red Planet.",
                "Jupiter is the largest planet in the solar system.",
                "Red is a color in the visible spectrum.",
            ],
        ),
        (
            "which language is primarily used for ios app development",
            [
                "Swift is the primary language for modern iOS app development.",
                "Kotlin is commonly used for Android development.",
                "Apple is a technology company based in Cupertino.",
            ],
        ),
    ]

    payloads = []
    for _ in range(args.rounds):
        for query, documents in cases:
            payloads.append(
                {
                    "model": args.model,
                    "query": query,
                    "documents": documents,
                    "top_n": 3,
                }
            )

    def one(index: int, payload: dict[str, object]) -> tuple[int, float, int, str]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        with request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read())
        elapsed_ms = (time.perf_counter() - started) * 1000
        return index, elapsed_ms, data["results"][0]["index"], str(payload["query"])

    latencies_ms: list[float] = []
    started = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for index, elapsed_ms, top_index, query in executor.map(
            lambda item: one(item[0], item[1]),
            enumerate(payloads, 1),
        ):
            latencies_ms.append(elapsed_ms)
            print(f"#{index} {elapsed_ms:.2f} ms top_index={top_index} query={query}")
    total_wall_s = time.perf_counter() - started

    ordered = sorted(latencies_ms)
    p50 = statistics.median(ordered)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    p95 = ordered[p95_index]
    print("--- summary ---")
    print(
        json.dumps(
            {
                "requests": len(ordered),
                "concurrency": args.concurrency,
                "rounds": args.rounds,
                "wall_s": round(total_wall_s, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "min_ms": round(ordered[0], 2),
                "max_ms": round(ordered[-1], 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
