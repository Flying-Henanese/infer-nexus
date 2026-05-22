#!/usr/bin/env python3
"""Minimal direct vLLM check for Qwen3 reranker behavior."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly initialize vLLM for Qwen3 reranker and run a sample score request."
    )
    parser.add_argument("--model", required=True, help="Local model path or HF model id.")
    parser.add_argument("--chat-template", required=True, help="Absolute path to reranker Jinja template.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument(
        "--query",
        default="what is the capital of france",
        help="Query used for the sanity-check rerank request.",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        default=[
            "Paris is the capital of France.",
            "Python is a programming language.",
            "Berlin is the capital of Germany.",
        ],
        help="Documents used for the sanity-check rerank request.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        import vllm
        from vllm import LLM
    except Exception as exc:  # pragma: no cover - debug script
        print(f"failed to import vllm: {exc}", file=sys.stderr)
        return 1

    print(f"vllm_version={getattr(vllm, '__version__', 'unknown')}")

    llm_kwargs = {
        "model": args.model,
        "task": "score",
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "chat_template": args.chat_template,
        "hf_overrides": {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    }

    print("llm_kwargs=" + json.dumps(llm_kwargs, ensure_ascii=False, indent=2))

    try:
        llm = LLM(**llm_kwargs)
    except Exception as exc:  # pragma: no cover - debug script
        print(f"llm_init_failed: {exc}", file=sys.stderr)
        return 2

    print(f"supported_tasks={getattr(llm, 'supported_tasks', None)}")

    try:
        outputs = llm.score(args.query, args.documents)
    except Exception as exc:  # pragma: no cover - debug script
        print(f"score_failed: {exc}", file=sys.stderr)
        return 3

    print("query=" + args.query)
    print("documents=" + json.dumps(args.documents, ensure_ascii=False, indent=2))
    print("scores=")
    for index, item in enumerate(outputs):
        score = getattr(getattr(item, "outputs", None), "score", None)
        print(json.dumps({"index": index, "document": args.documents[index], "score": score}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
