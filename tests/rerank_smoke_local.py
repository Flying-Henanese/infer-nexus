#!/usr/bin/env python3
"""Sequential local smoke test for rerank models."""

from __future__ import annotations

import argparse
import json
import time
from urllib import request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sequential local rerank smoke tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="bge-reranker")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    url = f"{args.base_url.rstrip('/')}/v1/rerank"
    headers = {"Content-Type": "application/json"}

    payloads = [
        {
            "model": args.model,
            "query": "what is the capital of france",
            "documents": [
                "Paris is the capital of France.",
                "Python is a programming language.",
                "Berlin is the capital of Germany.",
            ],
            "top_n": 3,
        },
        {
            "model": args.model,
            "query": "who wrote pride and prejudice",
            "documents": [
                "Jane Austen wrote Pride and Prejudice.",
                "Pride and Prejudice is a novel.",
                "Shakespeare wrote Hamlet.",
            ],
            "top_n": 3,
        },
        {
            "model": args.model,
            "query": "what planet is known as the red planet",
            "documents": [
                "Mars is known as the Red Planet.",
                "Jupiter is the largest planet in the solar system.",
                "Red is a color in the visible spectrum.",
            ],
            "top_n": 3,
        },
        {
            "model": args.model,
            "query": "which language is primarily used for ios app development",
            "documents": [
                "Swift is the primary language for modern iOS app development.",
                "Kotlin is commonly used for Android development.",
                "Apple is a technology company based in Cupertino.",
            ],
            "top_n": 3,
        },
        {
            "model": args.model,
            "query": "what is photosynthesis",
            "documents": [
                "Photosynthesis is the process by which plants convert light into chemical energy.",
                "Basketball is played with a ball and hoop.",
                "Plants need sunlight and water to grow.",
            ],
            "top_n": 3,
        },
        {
            "model": args.model,
            "query": "what does cpu stand for",
            "documents": [
                "CPU stands for Central Processing Unit.",
                "GPU stands for Graphics Processing Unit.",
                "RAM is a type of computer memory.",
            ],
            "top_n": 3,
        },
        {
            "model": args.model,
            "query": "what causes rainbows",
            "documents": [
                "Rainbows are caused by the refraction, dispersion, and reflection of light in water droplets.",
                "Thunder is the sound caused by lightning.",
                "A prism can split white light into colors.",
            ],
            "top_n": 3,
        },
        {
            "model": args.model,
            "query": "what is the boiling point of water",
            "documents": [
                "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                "Ice melts at 0 degrees Celsius.",
                "Steam is water vapor.",
            ],
            "top_n": 3,
        },
    ]

    for index, payload in enumerate(payloads, 1):
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        with request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read())
        elapsed_ms = (time.perf_counter() - started) * 1000
        top = data["results"][0]
        print(
            f"#{index} {elapsed_ms:.2f} ms "
            f"top_index={top['index']} "
            f"score={top['relevance_score']:.6f} "
            f"tokens={data['usage']['total_tokens']} "
            f"query={payload['query']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
