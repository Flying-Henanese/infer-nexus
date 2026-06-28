"""Built-in benchmark workloads."""

from infer_nexus.benchmark.config import WorkloadItem


PREFILL_CONTEXT_BLOCK = (
    "queue latency ttft tpot kv cache token throughput scheduler pressure "
    "prefix reuse admission control batch planning memory bandwidth. "
)


BUILTIN_WORKLOADS: dict[str, list[WorkloadItem]] = {
    "short_chat": [
        WorkloadItem(
            messages=[
                {"role": "user", "content": "Give a concise summary of infer-nexus."},
            ],
            max_tokens=128,
        )
    ],
    "rag_prompt": [
        WorkloadItem(
            messages=[
                {
                    "role": "system",
                    "content": "Answer using only the provided context.",
                },
                {
                    "role": "user",
                    "content": (
                        "Context: infer-nexus is a FastAPI gateway for model serving. "
                        "Question: what should client-side benchmarks measure?"
                    ),
                },
            ],
            max_tokens=256,
        )
    ],
    "long_context": [
        WorkloadItem(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Read the following repeated operational notes and summarize "
                        "the bottleneck signals to inspect: "
                        + "queue latency, TTFT, TPOT, KV cache usage, token throughput. " * 80
                    ),
                }
            ],
            max_tokens=512,
        )
    ],
    "prefill_intensive": [
        WorkloadItem(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are benchmarking a model with max_model_length 30000. "
                        "Read the following synthetic operations log and return only "
                        "a compact summary of the dominant bottleneck signals:\n\n"
                        + PREFILL_CONTEXT_BLOCK * 1500
                    ),
                }
            ],
            max_tokens=256,
        )
    ],
    "decode_intensive": [
        WorkloadItem(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are benchmarking decode throughput on a model with "
                        "max_model_length 30000. Write a long, detailed operational "
                        "analysis with numbered sections, concrete examples, and "
                        "continued elaboration until the response budget is exhausted."
                    ),
                }
            ],
            max_tokens=12_000,
        )
    ],
}


def get_builtin_workload(name: str) -> list[WorkloadItem]:
    """Return a built-in workload by name."""
    try:
        return BUILTIN_WORKLOADS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BUILTIN_WORKLOADS))
        raise ValueError(f"unknown workload {name!r}; available: {available}") from exc
