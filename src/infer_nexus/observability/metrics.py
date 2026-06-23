"""Prometheus 指标与可观测性快照。"""

from dataclasses import dataclass
from typing import Any

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ModuleNotFoundError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        """在依赖未安装的开发环境中保持应用可导入。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """接受真实 Prometheus collector 的构造参数。"""

        def labels(self, **_labels: str) -> "_NoopMetric":
            """返回自身以兼容 Prometheus collector 链式调用。"""
            return self

        def inc(self, _amount: float = 1.0) -> None:
            """忽略计数递增。"""

        def dec(self, _amount: float = 1.0) -> None:
            """忽略 Gauge 递减。"""

        def observe(self, _amount: float) -> None:
            """忽略直方图观测。"""

        def set(self, _value: float) -> None:
            """No-op Gauge setter."""

    Counter = Gauge = Histogram = _NoopMetric

    def generate_latest() -> bytes:
        """返回可读提示，正式环境应安装 prometheus-client。"""
        return b"# prometheus_client is not installed; metrics collectors are disabled.\n"


@dataclass(slots=True)
class MetricsSnapshot:
    """负载快照（阶段一占位指标）。"""

    active_models: int
    status: str = "stub"
    message: str = "metrics backend not connected"


LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)
STREAM_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


class GatewayMetrics:
    """封装网关实时指标，避免业务路由直接依赖 Prometheus 细节。"""

    def __init__(self) -> None:
        """注册第一阶段网关指标。"""
        self.requests_total = Counter(
            "infer_nexus_requests_total",
            "Total OpenAI-compatible gateway requests.",
            ("model", "task", "endpoint", "status"),
        )
        self.request_latency_seconds = Histogram(
            "infer_nexus_request_latency_seconds",
            "Gateway request latency before the response object is returned.",
            ("model", "task", "endpoint"),
            buckets=LATENCY_BUCKETS,
        )
        self.inflight_requests = Gauge(
            "infer_nexus_inflight_requests",
            "OpenAI-compatible gateway requests currently being handled.",
            ("model", "task", "endpoint"),
        )
        self.errors_total = Counter(
            "infer_nexus_errors_total",
            "Gateway errors grouped by stable error code.",
            ("model", "task", "code"),
        )
        self.admission_rejections_total = Counter(
            "infer_nexus_admission_rejections_total",
            "Requests rejected by admission control.",
            ("model", "reason"),
        )
        self.stream_ttft_seconds = Histogram(
            "infer_nexus_stream_ttft_seconds",
            "Streaming chat time to first emitted SSE chunk.",
            ("model",),
            buckets=STREAM_BUCKETS,
        )
        self.stream_chunk_interval_seconds = Histogram(
            "infer_nexus_stream_chunk_interval_seconds",
            "Interval between emitted streaming SSE chunks.",
            ("model",),
            buckets=STREAM_BUCKETS,
        )
        self.stream_tpot_seconds = Histogram(
            "infer_nexus_stream_tpot_seconds",
            "Chunk-level streaming TPOT approximation between emitted SSE chunks.",
            ("model",),
            buckets=STREAM_BUCKETS,
        )
        self.stream_completions_total = Counter(
            "infer_nexus_stream_completions_total",
            "Streaming chat responses grouped by terminal status.",
            ("model", "status"),
        )
        self.request_queue_seconds = Histogram(
            "infer_nexus_request_queue_seconds",
            "Gateway-observed request queue time when available.",
            ("model",),
            buckets=LATENCY_BUCKETS,
        )
        self.input_tokens_total = Counter(
            "infer_nexus_input_tokens_total",
            "Input tokens reported by model responses when available.",
            ("model", "task"),
        )
        self.output_tokens_total = Counter(
            "infer_nexus_output_tokens_total",
            "Output tokens reported by model responses when available.",
            ("model", "task"),
        )
        self.gateway_worker_inflight = Gauge(
            "infer_nexus_gateway_worker_inflight",
            "Requests currently admitted by this gateway worker process.",
            ("worker",),
        )
        self.gateway_worker_capacity = Gauge(
            "infer_nexus_gateway_worker_capacity",
            "Configured active-request capacity of this gateway worker process; zero is unlimited.",
            ("worker",),
        )
        self.gateway_worker_rejections_total = Counter(
            "infer_nexus_gateway_worker_rejections_total",
            "Requests rejected by process-local gateway worker admission.",
            ("worker", "reason"),
        )
        self.model_inflight = Gauge(
            "infer_nexus_model_inflight",
            "Requests currently admitted by this gateway worker for one model.",
            ("model",),
        )
        self.model_queue_depth = Gauge(
            "infer_nexus_model_queue_depth",
            "Requests waiting in the bounded gateway-local model queue.",
            ("model",),
        )
        self.runtime_guard_rejections_total = Counter(
            "infer_nexus_runtime_guard_rejections_total",
            "Requests rejected by gateway-local runtime guards.",
            ("model", "reason"),
        )
        self.admission_wait_seconds = Histogram(
            "infer_nexus_admission_wait_seconds",
            "Time spent waiting for a gateway-local runtime admission slot.",
            ("model",),
            buckets=LATENCY_BUCKETS,
        )
        self.serve_handle_calls_total = Counter(
            "infer_nexus_serve_handle_calls_total",
            "Ray Serve handle calls grouped by method and terminal status.",
            ("model", "method", "status"),
        )
        self.serve_handle_latency_seconds = Histogram(
            "infer_nexus_serve_handle_latency_seconds",
            "Ray Serve handle call latency observed by the gateway.",
            ("model", "method"),
            buckets=LATENCY_BUCKETS,
        )
        self.serve_handle_timeouts_total = Counter(
            "infer_nexus_serve_handle_timeouts_total",
            "Ray Serve handle calls or streams that timed out.",
            ("model", "method"),
        )
        self.serve_circuit_state = Gauge(
            "infer_nexus_serve_circuit_state",
            "Serve circuit breaker state for a model, where 1 means open.",
            ("model",),
        )

    def inc_inflight(self, *, model: str, task: str, endpoint: str) -> None:
        """记录一个正在处理的请求。"""
        self.inflight_requests.labels(model=model, task=task, endpoint=endpoint).inc()

    def dec_inflight(self, *, model: str, task: str, endpoint: str) -> None:
        """结束一个正在处理的请求。"""
        self.inflight_requests.labels(model=model, task=task, endpoint=endpoint).dec()

    def observe_request(
        self,
        *,
        model: str,
        task: str,
        endpoint: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """记录请求数量和网关延迟。"""
        self.requests_total.labels(
            model=model,
            task=task,
            endpoint=endpoint,
            status=status,
        ).inc()
        self.request_latency_seconds.labels(
            model=model,
            task=task,
            endpoint=endpoint,
        ).observe(max(latency_seconds, 0.0))

    def observe_error(self, *, model: str, task: str, code: str) -> None:
        """按稳定错误码记录失败请求。"""
        self.errors_total.labels(model=model, task=task, code=code).inc()

    def observe_admission_rejection(self, *, model: str, reason: str) -> None:
        """记录准入控制拒绝原因。"""
        self.admission_rejections_total.labels(model=model, reason=reason).inc()

    def observe_stream_ttft(self, *, model: str, seconds: float) -> None:
        """记录流式响应首块延迟。"""
        self.stream_ttft_seconds.labels(model=model).observe(max(seconds, 0.0))

    def observe_stream_chunk_interval(self, *, model: str, seconds: float) -> None:
        """记录流式响应相邻输出块间隔。"""
        value = max(seconds, 0.0)
        self.stream_chunk_interval_seconds.labels(model=model).observe(value)
        self.stream_tpot_seconds.labels(model=model).observe(value)

    def observe_stream_completion(self, *, model: str, status: str) -> None:
        """记录流式响应终止状态。"""
        self.stream_completions_total.labels(model=model, status=status).inc()

    def observe_request_queue(self, *, model: str, seconds: float) -> None:
        """记录请求排队耗时；当前仅在调用方有可靠队列时间时使用。"""
        self.request_queue_seconds.labels(model=model).observe(max(seconds, 0.0))

    def set_gateway_worker_inflight(self, *, worker: str, value: int) -> None:
        """Record requests admitted by this gateway worker process."""
        self.gateway_worker_inflight.labels(worker=worker).set(max(value, 0))

    def set_gateway_worker_capacity(self, *, worker: str, value: int) -> None:
        """Record the configured process-local active-request capacity."""
        self.gateway_worker_capacity.labels(worker=worker).set(max(value, 0))

    def observe_gateway_worker_rejection(self, *, worker: str, reason: str) -> None:
        """Record a process-local fail-fast rejection."""
        self.gateway_worker_rejections_total.labels(worker=worker, reason=reason).inc()

    def set_model_inflight(self, *, model: str, value: int) -> None:
        """Record active requests for one model inside this gateway worker."""
        self.model_inflight.labels(model=model).set(max(value, 0))

    def set_model_queue_depth(self, *, model: str, value: int) -> None:
        """Record bounded gateway-local queue depth for one model."""
        self.model_queue_depth.labels(model=model).set(max(value, 0))

    def observe_runtime_guard_rejection(self, *, model: str, reason: str) -> None:
        """Record a request rejected before it enters Ray Serve routing."""
        self.runtime_guard_rejections_total.labels(model=model, reason=reason).inc()

    def observe_admission_wait(self, *, model: str, seconds: float) -> None:
        """Record how long a request waited for a local admission slot."""
        self.admission_wait_seconds.labels(model=model).observe(max(seconds, 0.0))

    def observe_serve_handle_call(
        self,
        *,
        model: str,
        method: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """Record one Ray Serve handle call and its gateway-observed latency."""
        self.serve_handle_calls_total.labels(
            model=model,
            method=method,
            status=status,
        ).inc()
        self.serve_handle_latency_seconds.labels(
            model=model,
            method=method,
        ).observe(max(latency_seconds, 0.0))

    def observe_serve_handle_timeout(self, *, model: str, method: str) -> None:
        """Record a Ray Serve handle timeout or stream idle timeout."""
        self.serve_handle_timeouts_total.labels(model=model, method=method).inc()

    def set_serve_circuit_state(self, *, model: str, open: bool) -> None:
        """Record whether a model-level Serve circuit breaker is open."""
        self.serve_circuit_state.labels(model=model).set(1 if open else 0)

    def observe_token_usage(self, *, model: str, task: str, usage: Any) -> None:
        """在响应包含 usage 时记录输入和输出 token。"""
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            self.input_tokens_total.labels(model=model, task=task).inc(prompt_tokens)
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            self.output_tokens_total.labels(model=model, task=task).inc(completion_tokens)


GATEWAY_METRICS = GatewayMetrics()


def render_prometheus_metrics() -> tuple[bytes, str]:
    """生成 Prometheus 文本格式指标内容。"""
    return generate_latest(), CONTENT_TYPE_LATEST
