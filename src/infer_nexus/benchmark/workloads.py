"""Built-in benchmark workloads with cache-safe long-context variation.

The ``long_context`` workload deliberately puts a unique marker at the start of
each user message. Prefix caching only reuses a contiguous prefix, so changing
the first user-content block prevents the large repeated body from producing an
unrealistically high cache hit rate. A small shared chat-template prefix may
still be cached, which is representative of normal production traffic.

Optional environment variables:

``INFER_NEXUS_WORKLOAD_RUN_ID``
    Identifier used to derive per-request markers. When omitted, a new random
    value is generated whenever the benchmark process starts. Set it explicitly
    only when exact workload reproducibility is required.

``INFER_NEXUS_LONG_CONTEXT_VARIANTS``
    Number of unique long-context requests generated before the workload wraps.
    Default: 512. Keep this at least as large as ``--num-requests``.

``INFER_NEXUS_LONG_CONTEXT_OUTPUT_TOKENS``
    Center of the per-request ``max_tokens`` distribution. Default: 1000.

``INFER_NEXUS_LONG_CONTEXT_OUTPUT_JITTER``
    Maximum variation above or below the output-token center. Default: 200,
    producing an 800-1200 range with the default center.

``INFER_NEXUS_LONG_CONTEXT_MIN_REPEATS`` / ``MAX_REPEATS``
    Input-context repetition range. Defaults: 60 and 100.
"""

from __future__ import annotations

import hashlib
import os
import random
import secrets

from infer_nexus.benchmark.config import WorkloadItem


def _positive_int_env(name: str, default: int) -> int:
    """Read one positive integer environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def _non_negative_int_env(name: str, default: int) -> int:
    """Read one non-negative integer environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


PREFILL_CONTEXT_BLOCK = (
    "queue latency ttft tpot kv cache token throughput scheduler pressure "
    "prefix reuse admission control batch planning memory bandwidth. "
)

LONG_CONTEXT_BLOCKS = [
    (
        "queue latency ttft tpot kv cache token throughput scheduler pressure "
        "prefix reuse admission control batch planning memory bandwidth. "
    ),
    (
        "connection pool timeout retry backoff circuit breaker rate limiter "
        "load balancer health check graceful degradation failover clustering. "
    ),
    (
        "memory allocation garbage collection heap size stack overflow "
        "thread pool async io event loop coroutine task scheduler. "
    ),
    (
        "disk io throughput iops latency read write cache eviction policy "
        "buffer pool page cache write ahead log compaction. "
    ),
    (
        "network bandwidth packet loss retransmission window size "
        "congestion control flow control backpressure throttle. "
    ),
    (
        "cpu utilization instruction cache branch prediction pipeline "
        "speculative execution context switch interrupt handling. "
    ),
    (
        "authentication authorization encryption decryption key rotation "
        "token validation session management cookie handling. "
    ),
    (
        "index btree hash map bloom filter skip list radix tree "
        "query optimizer statistics histogram cardinality estimation. "
    ),
]

LONG_CONTEXT_QUESTIONS = [
    "summarize the key bottleneck signals mentioned in these operational notes",
    "identify the top three performance issues from these system metrics",
    "extract all references to latency and throughput measurements",
    "list the resource management concerns described in these logs",
    "find the scheduling and queuing related problems mentioned",
    "highlight the memory and cache related observations",
    "note any network and IO performance indicators",
    "compile a list of all system pressure points described",
]

SHORT_CHAT_PROMPTS = [
    "Give a concise summary of infer-nexus.",
    "Explain what Ray Serve does in this architecture.",
    "What are the main components of the gateway?",
    "How does vLLM integrate with the system?",
    "Describe the model loading process.",
    "What metrics should be monitored during inference?",
    "Explain the difference between streaming and non-streaming responses.",
    "What is the purpose of the admission control system?",
]

RAG_CONTEXTS = [
    (
        "infer-nexus is a FastAPI gateway for model serving.",
        "what should client-side benchmarks measure?",
    ),
    (
        "Ray Serve handles deployment lifecycle and autoscaling.",
        "how does Ray Serve manage model replicas?",
    ),
    (
        "vLLM provides the default inference backend.",
        "what are vLLM's main performance characteristics?",
    ),
    (
        "The model catalog stores registered model configurations.",
        "how are models registered and discovered?",
    ),
    (
        "OpenAI-compatible APIs allow easy client integration.",
        "what OpenAI endpoints are supported?",
    ),
    (
        "Accelerator pools share GPU resources across models.",
        "how is GPU memory managed?",
    ),
]

PREFILL_VARIATIONS = [
    "You are benchmarking a model with max_model_length 30000.",
    "You are evaluating a 70B parameter model with 32K context window.",
    "You are testing a fine-tuned model for code generation.",
    "You are stress-testing a multi-modal model with image inputs.",
    "You are measuring throughput on a quantized model deployment.",
    "You are profiling a model serving mixed workloads simultaneously.",
]


LONG_CONTEXT_VARIANTS = _positive_int_env(
    "INFER_NEXUS_LONG_CONTEXT_VARIANTS", 512
)
LONG_CONTEXT_OUTPUT_TOKENS = _positive_int_env(
    "INFER_NEXUS_LONG_CONTEXT_OUTPUT_TOKENS", 1000
)
LONG_CONTEXT_OUTPUT_JITTER = _non_negative_int_env(
    "INFER_NEXUS_LONG_CONTEXT_OUTPUT_JITTER", 200
)
LONG_CONTEXT_MIN_REPEATS = _positive_int_env(
    "INFER_NEXUS_LONG_CONTEXT_MIN_REPEATS", 60
)
LONG_CONTEXT_MAX_REPEATS = _positive_int_env(
    "INFER_NEXUS_LONG_CONTEXT_MAX_REPEATS", 100
)

if LONG_CONTEXT_MIN_REPEATS > LONG_CONTEXT_MAX_REPEATS:
    raise ValueError(
        "INFER_NEXUS_LONG_CONTEXT_MIN_REPEATS must not exceed "
        "INFER_NEXUS_LONG_CONTEXT_MAX_REPEATS"
    )
if LONG_CONTEXT_OUTPUT_TOKENS <= LONG_CONTEXT_OUTPUT_JITTER:
    raise ValueError(
        "INFER_NEXUS_LONG_CONTEXT_OUTPUT_TOKENS must be greater than "
        "INFER_NEXUS_LONG_CONTEXT_OUTPUT_JITTER"
    )

# A new process gets a new run id by default, preventing a previous benchmark
# run from warming the next run's long-context prefixes.
WORKLOAD_RUN_ID = os.environ.get("INFER_NEXUS_WORKLOAD_RUN_ID") or secrets.token_hex(16)


def _stable_seed(run_id: str) -> int:
    """Derive a repeatable RNG seed from a run identifier."""
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _request_marker(run_id: str, index: int) -> str:
    """Return a marker that differs before the large shared prompt body."""
    digest = hashlib.sha256(f"{run_id}:{index}".encode("utf-8")).hexdigest()
    return digest


def _generate_long_context_messages(
    *,
    marker: str,
    primary_block: str,
    secondary_block: str,
    question: str,
    repeat_count: int,
    target_output_tokens: int,
) -> list[dict[str, str]]:
    """Generate one long-context request with an early unique prefix."""
    # The marker intentionally appears before common instruction text. Adding a
    # unique suffix at the end would still allow nearly the entire prompt to hit
    # the prefix cache and would not serve this benchmark goal.
    content = (
        f"{marker}\n"
        "Read the operational notes below. "
        f"Your task is to {question}. "
        f"Produce a detailed report of approximately {target_output_tokens} tokens, "
        "using numbered sections, concrete evidence, and explicit conclusions. "
        "Continue until the requested level of detail has been covered.\n\n"
        + primary_block * repeat_count
        + secondary_block * max(1, repeat_count // 4)
    )
    return [{"role": "user", "content": content}]


def _generate_short_chat_messages(index: int) -> list[dict[str, str]]:
    prompt = SHORT_CHAT_PROMPTS[index % len(SHORT_CHAT_PROMPTS)]
    return [{"role": "user", "content": prompt}]


def _generate_rag_messages(index: int) -> list[dict[str, str]]:
    context, question = RAG_CONTEXTS[index % len(RAG_CONTEXTS)]
    return [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Context: {context} Question: {question}"},
    ]


def _generate_prefill_messages(index: int) -> list[dict[str, str]]:
    variation = PREFILL_VARIATIONS[index % len(PREFILL_VARIATIONS)]
    return [
        {
            "role": "user",
            "content": (
                f"{variation} "
                "Read the following synthetic operations log and return only "
                "a compact summary of the dominant bottleneck signals:\n\n"
                + PREFILL_CONTEXT_BLOCK * 1500
            ),
        }
    ]


def _generate_decode_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You are benchmarking decode throughput on a model with "
                "max_model_length 30000. Write a long, detailed operational "
                "analysis with numbered sections, concrete examples, and "
                "continued elaboration until the response budget is exhausted."
            ),
        }
    ]


def _build_long_context_workload() -> list[WorkloadItem]:
    """Build a large, cache-safe, reproducible-within-run workload pool."""
    rng = random.Random(_stable_seed(WORKLOAD_RUN_ID))
    minimum_output = LONG_CONTEXT_OUTPUT_TOKENS - LONG_CONTEXT_OUTPUT_JITTER
    maximum_output = LONG_CONTEXT_OUTPUT_TOKENS + LONG_CONTEXT_OUTPUT_JITTER
    items: list[WorkloadItem] = []

    for index in range(LONG_CONTEXT_VARIANTS):
        primary_index = rng.randrange(len(LONG_CONTEXT_BLOCKS))
        secondary_index = rng.randrange(len(LONG_CONTEXT_BLOCKS) - 1)
        if secondary_index >= primary_index:
            secondary_index += 1

        repeat_count = rng.randint(
            LONG_CONTEXT_MIN_REPEATS,
            LONG_CONTEXT_MAX_REPEATS,
        )
        max_tokens = rng.randint(minimum_output, maximum_output)
        question = rng.choice(LONG_CONTEXT_QUESTIONS)
        marker = _request_marker(WORKLOAD_RUN_ID, index)

        items.append(
            WorkloadItem(
                messages=_generate_long_context_messages(
                    marker=marker,
                    primary_block=LONG_CONTEXT_BLOCKS[primary_index],
                    secondary_block=LONG_CONTEXT_BLOCKS[secondary_index],
                    question=question,
                    repeat_count=repeat_count,
                    target_output_tokens=max_tokens,
                ),
                max_tokens=max_tokens,
            )
        )
    return items


def _build_short_chat_workload() -> list[WorkloadItem]:
    return [
        WorkloadItem(
            messages=_generate_short_chat_messages(index),
            max_tokens=128,
        )
        for index in range(len(SHORT_CHAT_PROMPTS))
    ]


def _build_rag_workload() -> list[WorkloadItem]:
    return [
        WorkloadItem(
            messages=_generate_rag_messages(index),
            max_tokens=256,
        )
        for index in range(len(RAG_CONTEXTS))
    ]


def _build_prefill_workload() -> list[WorkloadItem]:
    return [
        WorkloadItem(
            messages=_generate_prefill_messages(index),
            max_tokens=256,
        )
        for index in range(len(PREFILL_VARIATIONS))
    ]


def _build_decode_workload() -> list[WorkloadItem]:
    return [
        WorkloadItem(
            messages=_generate_decode_messages(),
            max_tokens=12_000,
        )
    ]


BUILTIN_WORKLOADS: dict[str, list[WorkloadItem]] = {
    "short_chat": _build_short_chat_workload(),
    "rag_prompt": _build_rag_workload(),
    "long_context": _build_long_context_workload(),
    "prefill_intensive": _build_prefill_workload(),
    "decode_intensive": _build_decode_workload(),
}


def get_builtin_workload(name: str) -> list[WorkloadItem]:
    """Return a built-in workload by name."""
    try:
        return BUILTIN_WORKLOADS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BUILTIN_WORKLOADS))
        raise ValueError(f"unknown workload {name!r}; available: {available}") from exc
