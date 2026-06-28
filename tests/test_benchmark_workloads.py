"""Built-in benchmark workload tests."""

from infer_nexus.benchmark.tokenizer import EstimatedTokenizer
from infer_nexus.benchmark.workloads import get_builtin_workload


def test_prefill_intensive_workload_has_long_input_and_short_output() -> None:
    """Prefill workload should spend most token budget before first token."""
    workload = get_builtin_workload("prefill_intensive")
    item = workload[0]
    input_tokens = EstimatedTokenizer().count_messages(item.messages)

    assert 25_000 <= input_tokens <= 29_500
    assert item.max_tokens == 256
    assert input_tokens > item.max_tokens * 90


def test_decode_intensive_workload_has_short_input_and_long_output_budget() -> None:
    """Decode workload should spend most token budget after first token."""
    workload = get_builtin_workload("decode_intensive")
    item = workload[0]
    input_tokens = EstimatedTokenizer().count_messages(item.messages)

    assert input_tokens < 300
    assert item.max_tokens == 12_000
    assert item.max_tokens > input_tokens * 40
