#!/usr/bin/env bash
set -euo pipefail

METRICS_URL="${1:-http://127.0.0.1:8000/metrics}"
SAMPLE_SECONDS="${2:-1}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "错误：未找到 python3。" >&2
  exit 1
fi

python3 - "$METRICS_URL" "$SAMPLE_SECONDS" <<'PY'
from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request


metrics_url = sys.argv[1]
try:
    sample_seconds = float(sys.argv[2])
except ValueError:
    print("错误：采样间隔必须是数字。", file=sys.stderr)
    raise SystemExit(2)

if sample_seconds <= 0:
    print("错误：采样间隔必须大于 0。", file=sys.stderr)
    raise SystemExit(2)

metric_pattern = re.compile(
    r"^(vllm:[^{ ]+)(?:\{[^}]*\})?\s+([-+eE0-9.]+)$",
    re.MULTILINE,
)


def snapshot() -> dict[str, float]:
    request = urllib.request.Request(
        metrics_url,
        headers={"Accept": "text/plain"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法读取 {metrics_url}: {exc}") from exc

    result: dict[str, float] = {}
    for name, raw_value in metric_pattern.findall(text):
        result[name] = result.get(name, 0.0) + float(raw_value)
    if not result:
        raise RuntimeError(f"{metrics_url} 没有返回 vllm 指标")
    return result


def raw_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return max(0.0, after.get(name, 0.0) - before.get(name, 0.0))


def rate(
    before: dict[str, float],
    after: dict[str, float],
    name: str,
    elapsed: float,
) -> float:
    return raw_delta(before, after, name) / elapsed


def histogram_average(
    before: dict[str, float],
    after: dict[str, float],
    base: str,
    *,
    multiplier: float = 1.0,
) -> float | None:
    count = raw_delta(before, after, f"{base}_count")
    total = raw_delta(before, after, f"{base}_sum")
    return total / count * multiplier if count > 0 else None


def show_ms(value: float | None) -> str:
    return f"{value:8.2f} ms" if value is not None else "     N/A"


def show_number(value: float | None, unit: str = "") -> str:
    if value is None:
        return "     N/A"
    suffix = f" {unit}" if unit else ""
    return f"{value:8.2f}{suffix}"


try:
    previous = snapshot()
except RuntimeError as exc:
    print(f"错误：{exc}", file=sys.stderr)
    raise SystemExit(1)

previous_time = time.monotonic()

try:
    while True:
        time.sleep(sample_seconds)
        try:
            current = snapshot()
        except RuntimeError as exc:
            print(f"\033[2J\033[H错误：{exc}\n将在下一个采样周期重试。", flush=True)
            continue

        current_time = time.monotonic()
        elapsed = max(current_time - previous_time, 0.001)

        hits = rate(previous, current, "vllm:prefix_cache_hits_total", elapsed)
        queries = rate(previous, current, "vllm:prefix_cache_queries_total", elapsed)
        hit_rate = hits / queries * 100 if queries > 0 else None

        prompt_tps = rate(previous, current, "vllm:prompt_tokens_total", elapsed)
        generation_tps = rate(previous, current, "vllm:generation_tokens_total", elapsed)
        request_rps = rate(previous, current, "vllm:request_success_total", elapsed)
        preemptions = rate(previous, current, "vllm:num_preemptions_total", elapsed)

        ttft = histogram_average(
            previous, current, "vllm:time_to_first_token_seconds", multiplier=1000
        )
        itl = histogram_average(
            previous, current, "vllm:inter_token_latency_seconds", multiplier=1000
        )
        tpot = histogram_average(
            previous,
            current,
            "vllm:request_time_per_output_token_seconds",
            multiplier=1000,
        )
        queue_time = histogram_average(
            previous, current, "vllm:request_queue_time_seconds", multiplier=1000
        )
        prefill_time = histogram_average(
            previous, current, "vllm:request_prefill_time_seconds", multiplier=1000
        )
        inference_time = histogram_average(
            previous, current, "vllm:request_inference_time_seconds", multiplier=1000
        )
        e2e = histogram_average(
            previous, current, "vllm:e2e_request_latency_seconds", multiplier=1000
        )
        prefill_tokens = histogram_average(
            previous, current, "vllm:request_prefill_kv_computed_tokens"
        )
        iteration_tokens = histogram_average(
            previous, current, "vllm:iteration_tokens_total"
        )

        running = current.get("vllm:num_requests_running", 0.0)
        waiting = current.get("vllm:num_requests_waiting", 0.0)
        kv_usage = current.get("vllm:kv_cache_usage_perc", 0.0) * 100

        print("\033[2J\033[H", end="")
        print("========== vLLM 实时运行状态 ==========")
        print(f"采集地址                   指标来源               : {metrics_url}")
        print(f"采样窗口                   本次统计区间           : {elapsed:8.2f} s")
        print()
        print(
            "prefix_cache_hit_rate      Prefix Cache区间命中率 : "
            + (f"{hit_rate:8.2f} %" if hit_rate is not None else "     N/A")
        )
        print(f"prefix_cache_hits          每秒命中缓存Token数    : {hits:8.2f} token/s")
        print(f"prefix_cache_queries       每秒查询缓存Token数    : {queries:8.2f} token/s")
        print()
        print(f"prompt_tokens              Prefill输入吞吐量      : {prompt_tps:8.2f} token/s")
        print(f"generation_tokens          Decode输出吞吐量       : {generation_tps:8.2f} token/s")
        print(f"request_success            成功请求吞吐量         : {request_rps:8.2f} req/s")
        print()
        print(f"request_prefill_time       平均Prefill阶段耗时    : {show_ms(prefill_time)}")
        print(
            "prefill_computed_tokens    每请求实际计算KV Token : "
            + show_number(prefill_tokens, "token")
        )
        print(
            "iteration_tokens           每次引擎迭代Token数     : "
            + show_number(iteration_tokens, "token")
        )
        print(f"time_to_first_token        平均首Token延迟 TTFT   : {show_ms(ttft)}")
        print(f"inter_token_latency        平均Token间延迟 ITL    : {show_ms(itl)}")
        print(f"time_per_output_token      平均每输出Token耗时    : {show_ms(tpot)}")
        print(f"request_queue_time         平均请求排队延迟       : {show_ms(queue_time)}")
        print(f"request_inference_time     平均推理阶段耗时       : {show_ms(inference_time)}")
        print(f"e2e_request_latency        平均端到端请求延迟     : {show_ms(e2e)}")
        print()
        print(f"num_requests_running       正在运行的请求数       : {running:8.0f}")
        print(f"num_requests_waiting       正在等待的请求数       : {waiting:8.0f}")
        print(f"kv_cache_usage_perc        KV Cache使用率         : {kv_usage:8.2f} %")
        print(f"num_preemptions            每秒发生的抢占数       : {preemptions:8.2f} /s")
        print("\n按 Ctrl+C 退出。", flush=True)

        previous = current
        previous_time = current_time
except KeyboardInterrupt:
    print("\n监控已停止。")
PY
