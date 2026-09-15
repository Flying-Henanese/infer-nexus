# Ray Serve Hosted vLLM Monitoring

This runbook covers local `backend: vllm` models hosted by Ray Serve replicas.
It intentionally excludes `vllm_openai_proxy` models and standalone vLLM HTTP
servers.

Current validated state:

- infer-nexus gateway metrics are exposed from `http://<host>:8000/metrics`.
- Ray service-discovery is available from
  `http://127.0.0.1:8265/api/prometheus/sd` on the runtime host.
- Ray Serve deployment and replica metrics are visible from the Ray metrics
  target that belongs to the current Ray session. The target port changes after
  Ray restarts; do not hard-code it.
- Embedded vLLM internal metrics such as `vllm:num_requests_running` and
  `vllm:kv_cache_usage_perc` have not been observed through Ray metrics in the
  current local Ray Serve runtime shape.

## Scope

The monitoring target is the full local serving path:

```text
OpenAI-compatible request
-> infer-nexus gateway
-> RuntimeExecutor in serve mode
-> Ray Serve deployment and replica
-> replica-local vLLM runtime
-> accelerator hardware
```

Prometheus should scrape each layer directly where possible:

- infer-nexus gateway `/metrics`
- Ray metrics endpoint for Ray Serve deployment and replica state
- vLLM metrics only if the embedded runtime exposes them through the scraped Ray
  metrics source
- node/process/GPU/NPU exporters for hardware and process pressure

Do not make the gateway re-export Ray or vLLM metrics as the first step. Direct
scraping keeps ownership clear and avoids duplicated metric names.

## Quick Checks

Run these commands on the runtime host first.

Check the gateway metrics endpoint:

```bash
curl -sS http://127.0.0.1:8000/metrics | grep -E \
  'infer_nexus_requests_total|infer_nexus_request_latency_seconds|infer_nexus_errors_total|infer_nexus_stream_ttft_seconds|infer_nexus_stream_tpot_seconds|infer_nexus_request_queue_seconds'
```

Check Ray Serve application health:

```bash
serve status
```

Expected shape:

```text
applications:
  infer-nexus-model-...:
    status: RUNNING
    deployments:
      model-...:
        status: HEALTHY
        replica_states:
          RUNNING: 1
```

Check Ray Prometheus service discovery:

```bash
curl -sS http://127.0.0.1:8265/api/prometheus/sd
```

Example output:

```json
[{"labels": {"job": "ray"}, "targets": ["192.168.0.194:39601", "192.168.0.194:44217", "192.168.0.194:44227"]}]
```

The Ray Serve metrics target is dynamic. In one validated run, `39601` carried
Ray Serve metrics, while `44217` and `44227` carried autoscaler/dashboard
component metrics. Always use the current discovery output.

## Raw Metric Discovery

Before starting Prometheus, directly inspect each Ray target:

```bash
for target in $(python -c 'import json; print(" ".join(json.load(open("/tmp/ray/prom_metrics_service_discovery.json"))[0]["targets"]))'); do
  echo "===== $target ====="
  python scripts/monitoring/inspect_ray_serve_vllm_metrics.py \
    --metrics-url "http://$target/metrics" || true
done
```

If the discovery file is not present, use the Ray dashboard API instead:

```bash
python - <<'PY'
import json
import urllib.request

targets = json.load(urllib.request.urlopen("http://127.0.0.1:8265/api/prometheus/sd"))[0]["targets"]
print(" ".join(targets))
PY
```

Then run the inspection script against the listed targets.

For a fresh deployment, send one lightweight request first so request-scoped
metrics have samples:

```bash
curl -sS http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":"hello"}' >/dev/null

sleep 10
```

## Prometheus Scrape

Use the example at:

```text
monitoring/prometheus/ray-serve-vllm.yml
```

The example includes the Ray file service-discovery path observed in the current
environment:

```text
/tmp/ray/prom_metrics_service_discovery.json
```

Ray may also write service-discovery files under the active Ray session
directory. Check the actual target files on the runtime host before starting
Prometheus:

```bash
find /tmp/ray -name prometheus_sd.json \
  -o -name prometheus_service_discovery.json \
  -o -name prom_metrics_service_discovery.json
```

Update the `file_sd_configs` path in the example if the target file differs.

Start Prometheus after confirming the file exists:

```bash
prometheus --config.file=monitoring/prometheus/ray-serve-vllm.yml
```

Then verify Prometheus itself:

```bash
curl -sS http://127.0.0.1:9090/-/ready
curl -sS http://127.0.0.1:9090/api/v1/targets
```

## Metric Discovery

Run the inspection script against Prometheus when the Prometheus API is
reachable:

```bash
python scripts/monitoring/inspect_ray_serve_vllm_metrics.py \
  --prometheus-url http://127.0.0.1:9090
```

If Prometheus is not configured yet, run it against a raw Ray metrics endpoint:

```bash
python scripts/monitoring/inspect_ray_serve_vllm_metrics.py \
  --metrics-url http://127.0.0.1:<ray-metrics-port>/metrics
```

The first pass should confirm whether Ray Serve metrics are visible and whether
embedded vLLM metrics are visible from the same scrape source.

## Gateway Metrics

The gateway endpoint is:

```bash
curl -sS http://127.0.0.1:8000/metrics
```

Useful metrics:

- `infer_nexus_requests_total`: OpenAI-compatible gateway request count.
- `infer_nexus_request_latency_seconds`: gateway route latency histogram.
- `infer_nexus_inflight_requests`: requests currently handled by the gateway.
- `infer_nexus_errors_total`: gateway errors by stable code.
- `infer_nexus_admission_rejections_total`: admission-control rejections.
- `infer_nexus_stream_ttft_seconds`: gateway streaming time to first emitted
  SSE chunk.
- `infer_nexus_stream_tpot_seconds`: chunk-level streaming TPOT approximation.
- `infer_nexus_stream_chunk_interval_seconds`: compatibility name for the same
  chunk interval observation.
- `infer_nexus_stream_completions_total`: stream terminal status.
- `infer_nexus_request_queue_seconds`: defined for queue-time observations, but
  not currently connected to a reliable gateway-side queue-time source.
- `infer_nexus_input_tokens_total` and `infer_nexus_output_tokens_total`: token
  counts when the backend response contains usage.

## Primary Ray Serve Signals

Use Ray Serve metrics as the first queue-time source:

- `ray_serve_handle_request_counter_total`
- `ray_serve_num_router_requests_total`
- `ray_serve_deployment_queued_queries`
- `ray_serve_request_router_fulfillment_time_ms`
- `ray_serve_num_ongoing_requests_at_replicas`
- `ray_serve_replica_processing_queries`
- `ray_serve_replica_utilization_percent`
- `ray_serve_deployment_processing_latency_ms`
- `ray_serve_deployment_request_counter_total`
- `ray_serve_deployment_error_counter_total`
- `ray_serve_deployment_replica_healthy`
- `ray_serve_deployment_replica_starts_total`
- `ray_serve_autoscaling_target_replicas`
- `ray_serve_autoscaling_desired_replicas`
- `ray_serve_controller_num_control_loops`

These signals explain whether requests are waiting before replica execution,
whether replicas are saturated, and how long replica processing takes.

For local `DeploymentHandle` traffic, prioritize these panels first:

- queued queries by deployment
- ongoing requests at replicas
- replica processing queries
- replica utilization percent
- deployment processing latency
- handle request counter
- deployment request and error counters
- replica health

## vLLM Signals When Visible

If the embedded vLLM runtime exposes vLLM metrics through the scraped source,
add panels for:

- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:kv_cache_usage_perc`
- `vllm:request_queue_time_seconds`
- `vllm:time_to_first_token_seconds`
- `vllm:request_time_per_output_token_seconds`
- `vllm:prompt_tokens_total`
- `vllm:generation_tokens_total`

If these metrics are missing, do not assume vLLM is idle. The embedded runtime
may simply not expose the standalone vLLM server metrics path. In that case,
use Ray Serve queue/replica metrics plus hardware exporters first, then consider
adding infer-nexus replica-level custom metrics.

Current observed state: embedded vLLM metrics are missing from the Ray metrics
targets. This means KV cache usage, vLLM waiting/running requests, and vLLM
token-throughput counters are not currently available from Prometheus.

## Hardware Correlation

Correlate Ray Serve and vLLM signals with hardware exporter panels:

- GPU/NPU utilization
- GPU memory or Ascend HBM usage
- power and temperature
- gateway/Ray/vLLM process CPU and RSS

Typical interpretations:

- High Ray Serve queue with high hardware utilization: increase capacity or
  reduce workload pressure.
- High Ray Serve queue with low hardware utilization: inspect replica count,
  batching, routing, or runtime stalls.
- High vLLM waiting requests with high KV cache usage: tune concurrency, max
  sequence length, prefix cache behavior, or model placement.
- Low queue and low utilization: benchmark load is too light or request launch
  rate is too low.

If vLLM metrics are unavailable, replace the vLLM-specific checks with Ray Serve
replica metrics and hardware metrics:

- High Ray Serve queue and high replica utilization: model replicas are the
  immediate bottleneck.
- High Ray Serve queue and low replica utilization: inspect routing, request
  admission, runtime stalls, or too few active replicas.
- Low Ray Serve queue and low hardware utilization: increase benchmark
  concurrency or request rate before tuning model runtime parameters.

## Troubleshooting

If `127.0.0.1:9090` refuses connections, Prometheus is not running. Start it
with `monitoring/prometheus/ray-serve-vllm.yml`.

If `curl http://127.0.0.1:8265/api/prometheus/sd` works but a target refuses
connections, the discovery file may contain a stale target from a previous Ray
session. Refresh the target list and use the current addresses.

If a Ray target only shows `autoscaler_*` or `ray_component_*`, it is a Ray
control-plane target, not the Serve metrics target. Inspect all targets from
service discovery.

If `serve status` is healthy but no `ray_serve_*` metrics appear after a request,
confirm that the inspection script is the current version. The current script
prints four groups:

```text
Ray Serve routing metrics
Ray Serve replica metrics
Ray Serve lifecycle metrics
vLLM metrics
```

Older script output only shows `Ray Serve metrics` and `vLLM metrics`, and can
miss useful Serve indicators.

## Next Code Step

Only add infer-nexus replica-level metrics after metric discovery shows that
Ray Serve is visible but embedded vLLM internals are not. A focused custom
metric layer should record model, deployment, replica, status, replica latency,
stream TTFT, and stream chunk intervals. It should not try to emulate vLLM KV
cache or scheduler metrics.

## Structured Application Logs And Request IDs

Application logs complement these metrics; they explain individual request and
runtime events but do not replace aggregate counters, histograms, or Ray queue
signals. Request IDs are intentionally not Prometheus labels.

The checked-in profiles select human-readable console output in
`config/settings.yaml` and JSONL in the CUDA and Ascend Compose settings. The
Application records carry `event`, `request_id` when present, `model`, `task`,
`backend`, Serve app/deployment,
`outcome`, `status_code`, `duration_ms`, and stable `error_code` fields. Known
sensitive fields are redacted. Prompt and message bodies, embeddings, rerank
documents, credentials, and per-token/per-chunk records are excluded.

The Gateway selects an inbound validated `X-Request-ID`, falls back to the Ray
Serve request ID when available, then generates a UUID. Every response includes
`X-Request-ID`; streaming responses temporarily also include
`X-Infer-Nexus-Request-ID` for compatibility. That ID travels beside internal
Serve handle payloads and is bound by the model replica. Proxy upstream calls
include it when the model's `headers_policy.pass_request_id` is enabled.

Each HTTP request has one Gateway terminal event: `request.completed` or
`request.failed`. A streamed request also has one stream sub-lifecycle event:
`stream.completed`, `stream.failed`, or `stream.cancelled`. Stream failures own
their traceback; the request terminal event does not duplicate it. The
admission middleware owns `admission.rejected`, and model replicas own
`model.replica.initializing`, `model.replica.ready`, and
`model.replica.failed`. Runtime handle failures emit `runtime.call.failed`;
proxy upstream failures emit `proxy.request.failed` with safe host/status fields.
The request terminal event includes `upstream_status` when a proxy returns an
HTTP error. Successful `/healthz`, `/readyz`, and `/metrics` events are
suppressed; ordinary successful request events follow `success_sample_rate`.

## Log File Discovery

### Local bootstrap scripts

- `.infer-nexus/logs/ray_bootstrap.log` contains Ray CLI startup output.
- `.infer-nexus/logs/serve_runtime.log` contains the one-shot Serve deployer's
  output.
- Actual Ray session files are under
  `.infer-nexus/ray/session_latest/logs/`. When attached to an external Ray
  cluster, the startup script may instead report
  `/tmp/ray/session_latest/logs/`.
- Ray component files are configured for 50 MiB rotation with three backups.

The local profile uses console records, so search directly:

```bash
rg 'incident-2026-09-15-42' .infer-nexus/logs .infer-nexus/ray/session_latest/logs
```

### Docker Compose

CUDA and Ascend Compose export every service's logs into the repository-root
`logs/` directory. The first level is the Compose service; the final level
separates the container command log from Ray's session files:

```text
logs/<service>/container.log
logs/<service>/ray/session_latest/logs/
```

The one-shot `log-init` service creates the directories and grants the
non-root runtime user write access before `ray-head`, `ray-worker`, or
`serve-deployer` starts. Inspect service logs directly from the host with:

```bash
tail -F logs/ray-head/container.log
find logs/ray-worker/ray/session_latest/logs -maxdepth 2 -type f -print
rg 'incident-2026-09-15-42' logs
```

Use `docker compose -f ascend_deploy/docker-compose.yml ...` for Ascend Compose
commands. `container.log` combines the service command's stdout and stderr and
is append-only on the host. Docker's `json-file` rotation remains a fallback
for early container failures, while Ray component files use 50 MiB rotation
with three backups. Configure the host's `logrotate` if `container.log` needs a
retention limit.

For JSONL application records copied from a process log, filter using the
stable fields:

```bash
jq -c 'select(.request_id == "incident-2026-09-15-42")' gateway.jsonl
jq -c 'select(.event == "admission.rejected")' gateway.jsonl
jq -c 'select(.event == "model.replica.failed")' ray-worker.jsonl
```

Ray and vLLM framework records may use different JSON schemas. Do not assume
every third-party line has infer-nexus's `event` or request fields.

### Ray and Serve access logging

The Serve deployer configures Ray Core and Serve logging for the selected
console/JSON format, and sets vLLM to propagate records through the
process-owned application handler. The public Serve proxy's access logging is
controlled by `observability.logging.access_log`. Duplicate access logs on the
Gateway and model replica deployments are disabled. In deployed settings,
`uvicorn.access` is WARNING; in the local profile it is INFO. `RAY_BACKEND_LOG_JSON`
is enabled, with Ray rotation set to 50 MiB and three backups. vLLM's level
comes from `observability.logging.level`; request or prompt logging is not
enabled.

## Request-Centric Incident Checks

Start with a request's ID, then inspect the Gateway terminal record and, for
streams, the related stream terminal event. Query `outcome`, `status_code`,
`duration_ms`, `error_code`, `model`, `task`, and `deployment`; do not inspect
request content.

| Symptom | Log/metric signal | Interpretation |
|---|---|---|
| Process-local Gateway overload | `admission.rejected` with the same request ID; `infer_nexus_gateway_worker_rejections_total` | Worker admission rejected before route execution. Inspect `reason` and `retry_after_seconds`. |
| Serve ingress queue overload | Ray proxy access logs and `serve_deployment_queued_queries` | The request was rejected before FastAPI, so it may have no Gateway request event or generated Gateway ID. |
| Runtime timeout | Request terminal event with `outcome="timeout"`; matching Serve handle and deployment metrics | Inspect `error_code`, status, handle timeout counters, and model replica logs. |
| Proxy upstream error | `proxy.request.failed` with `upstream_host`, `upstream_status`, and safe error code | Correlate `request_id` with the Gateway terminal event and its `upstream_status`; do not log the full URL or payload. |
| Model startup failure | `model.replica.failed` with `backend_startup_failed` | Startup is not attached to an HTTP request; inspect the same replica's Ray logs. |
| Stream backend failure | `stream.failed`, then the request terminal event with the same ID | The stream event owns the traceback; the request event provides final HTTP outcome. |
| Client disconnect | `stream.cancelled`, then request outcome `cancelled` | The client ended the stream; investigate client/network behavior and do not expect a backend traceback. |

When debugging ingress queue pressure, use Ray Serve's dynamic metrics target
and `serve_deployment_queued_queries`; this outer rejection does not pass through
the application middleware. For end-to-end health, correlate the same
request ID across gateway and model replica records and use the existing
`/metrics` endpoint for aggregate request/stream metrics.
