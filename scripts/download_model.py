#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from infer_nexus.artifacts.download import (
    build_model_registration_snippet,
    prepare_huggingface_download,
)
from infer_nexus.core.config import load_settings
from infer_nexus.model_store import LocalModelStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model into the local infer-nexus model store.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example Qwen/Qwen3-32B-Instruct")
    parser.add_argument("--name", required=True, help="Stable infer-nexus model name")
    parser.add_argument("--alias", required=True, help="Client-facing model alias")
    parser.add_argument("--task", required=True, choices=["chat", "embedding", "rerank", "vlm"])
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--revision")
    parser.add_argument("--dtype")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cpu-per-replica", type=float, default=4)
    parser.add_argument("--gpu-per-replica", type=float, default=1)
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=1)
    parser.add_argument("--settings-path", default="config/settings.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings_path)
    model_store = LocalModelStore.from_settings(settings.model_store)

    downloaded_path = prepare_huggingface_download(
        model_store=model_store,
        repo_id=args.repo_id,
        revision=args.revision,
        endpoint=settings.model_store.huggingface_endpoint,
    )

    relative_model_path = downloaded_path.relative_to(model_store.root_dir)
    snippet = build_model_registration_snippet(
        name=args.name,
        alias=args.alias,
        task=args.task,
        backend=args.backend,
        model_path=relative_model_path.as_posix(),
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        cpu_per_replica=args.cpu_per_replica,
        gpu_per_replica=args.gpu_per_replica,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
    )

    print(f"Downloaded model to: {downloaded_path}")
    print("\nSuggested config/models.yaml entry:\n")
    print(snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
