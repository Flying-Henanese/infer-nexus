# infer-nexus 统一入口 + 单 upstream vLLM Proxy 实施清单

日期：2026-05-20  
适用分支：`refactor-per-model-serve-apps`  
目标读者：开发、测试、运维

---

## 1. 背景与目标

当前目标是让 `infer-nexus` 对外保持一个 OpenAI 兼容入口，同时允许部分模型通过上游 OpenAI-compatible vLLM 服务承载协议语义。

核心目标：

1. 对外只有统一入口：`/v1/*`。
2. 客户端只通过请求体里的 `model` 字段选择模型。
3. 模型实例仍由团队自托管，例如 Ray Serve + vLLM。
4. gateway 负责治理能力：鉴权、准入、路由、观测和网关阶段错误边界。
5. gateway 不重新实现 vLLM/OpenAI 协议细节，尽量透明转发。

本轮范围已经明确收敛为：**单 upstream proxy 强化**。

不在本轮实现 gateway 侧 upstream 实例池、熔断摘除、跨 upstream 负载均衡。若 upstream 由 Ray Serve 托管，模型副本池、健康检查、扩缩容和 replica 路由由 Ray Serve 负责。

---

## 2. 目标架构

```text
Client
  -> infer-nexus gateway (/v1/*)
      -> model routing by request.model
          -> one configured upstream OpenAI-compatible vLLM endpoint
              -> Ray Serve / vLLM owns replicas, health, and autoscaling
```

关键原则：

1. gateway 只做最小必要改写，通常只改写 `model` 到 `upstream_model_name`。
2. 请求体其他字段尽量原样透传。
3. 上游 HTTP 状态码、body、content type 尽量原样返回。
4. 流式场景做 SSE 字节透传，不解析事件、不重组 chunk。
5. 网关阶段失败才生成 infer-nexus 自己的 OpenAI-style JSON error。

---

## 3. 范围与非范围

### 3.1 本轮范围

1. 强化 `vllm_openai_proxy` 的单 upstream 非流式透传。
2. 强化 upstream 4xx/5xx 原样透传。
3. 强化 SSE 字节透传和 content type 保留。
4. 响应头增加 `X-Infer-Nexus-Request-ID`，便于关联日志和调用链。
5. 补充 pytest 脚本，方便在可运行环境验证。
6. 同步文档，删除 gateway upstream 实例池相关设计。

### 3.2 暂不纳入

1. gateway 侧多 upstream 实例池。
2. gateway 侧熔断、摘除、恢复窗口。
3. 动态模型注册。
4. 跨地域智能路由。
5. 全量新 API，例如 Responses API。
6. 强制把当前已跑通的 `mineru backend: vllm` 配置切到 proxy。

---

## 4. 文件级改动清单

### 4.1 Runtime 执行层

#### `src/infer_nexus/runtime/executor.py`

改动项：

1. `_proxy_payload` 继续保持最小改写策略：
   - 仅改写 `model`。
   - 其他字段保持原样。
2. 非流式 proxy 响应直接返回 Starlette `Response`：
   - 保留 upstream `status_code`。
   - 保留 upstream `body`。
   - 保留 upstream `content-type`。
   - 添加 `X-Infer-Nexus-Request-ID`。
3. upstream 4xx/5xx 不再被 gateway JSON 标准化，直接透传。
4. SSE streaming 保持字节透传：
   - 不解析事件。
   - 不重组 chunk。
   - 保留 `text/event-stream` content type。
   - 添加 `X-Infer-Nexus-Request-ID`。
5. 连接失败、连接超时、读取超时等没有 upstream HTTP 响应的情况，仍抛出 gateway-stage error。

验收点：

1. 同一请求经 gateway 与直连 upstream 的响应语义一致。
2. proxy 路径不再触发本地 `SamplingParams` 兼容问题。
3. 上游错误不会被 gateway 改写成另一种错误 body。

### 4.2 API 路由与错误语义层

#### `src/infer_nexus/api/openai_routes.py`

改动项：

1. 保持网关错误与上游错误分层：
   - 网关阶段失败：统一 OpenAI-style JSON error。
   - 上游阶段失败：由 proxy executor 原样透传。
2. `RuntimeNotConnectedError` 按错误码映射状态：
   - `upstream_timeout` -> `504 service_unavailable_error`。
   - `backend_misconfigured` -> `500 internal_server_error`。
   - `unsupported_parameter` -> `400 invalid_request_error`。
   - 其他未接通 runtime -> 保留 `501 not_implemented_error`。

### 4.3 配置层

#### `config/models.yaml`

本轮不强制修改当前已跑通的 `mineru backend: vllm` 配置。

如需启用单 upstream proxy，可按模型粒度切换：

```yaml
backend: vllm_openai_proxy
proxy_config:
  upstream_base_url: http://127.0.0.1:8001/v1
  upstream_model_name: opendatalab/MinerU2.5-2509-1.2B
  auth:
    mode: none
  timeout:
    connect_seconds: 3
    read_seconds: 180
    write_seconds: 30
    pool_seconds: 5
  retry:
    max_attempts: 1
    backoff_ms: 0
    retry_on_status: [502, 503, 504]
  streaming:
    enabled: true
    passthrough_sse: true
  headers_policy:
    pass_request_id: true
    forward_authorization: false
```

### 4.4 测试层

#### `tests/test_dispatcher.py`

覆盖：

1. proxy payload 除 `model` 外不改写。
2. 非流式成功响应透传。
3. upstream 4xx/5xx body/status/content type 透传。
4. embeddings/rerank proxy 路径也保持 Response passthrough。

#### `tests/test_proxy_streaming.py`

覆盖：

1. SSE `content-type` 保持。
2. SSE chunk 字节不改写。
3. streaming response 包含 `X-Infer-Nexus-Request-ID`。

#### `tests/test_api.py`

覆盖：

1. gateway 自己产生的 runtime timeout 映射为稳定 `504 upstream_timeout`。
2. 现有 unknown model、task mismatch、runtime not connected 等网关错误保持稳定。

---

## 5. 分阶段实施顺序

### Phase 1：单 upstream proxy 透传强化

1. 修改 `executor.py`。
2. 修改 `openai_routes.py` 错误映射。
3. 补充 `test_dispatcher.py`。
4. 新增 `test_proxy_streaming.py`。

完成标准：

1. 非流式 proxy 响应不再经过本地 response schema 重塑。
2. 上游错误 status/body 原样返回。
3. 响应包含 `X-Infer-Nexus-Request-ID`。
4. SSE 字节透传测试可在目标环境运行。

### Phase 2：文档同步

1. 删除设计文档中的 gateway upstream 实例池、熔断、摘除、恢复窗口描述。
2. 明确 Ray Serve 负责模型副本池和扩缩容。
3. 更新 README / MINIMAL_STARTUP 中的 proxy 行为说明。

完成标准：

1. 文档不再暗示 gateway 需要重复实现 Ray Serve 的 replica pool。
2. 新同学能理解 `vllm` 本地路径和 `vllm_openai_proxy` 单 upstream 路径的边界。

---

## 6. 最终验收标准

1. 对外仍只有 `/v1/*` 统一入口。
2. 所有模型仍通过 `request.model` 路由。
3. proxy 模型只改写 `model`，其余请求字段尽量原样透传。
4. upstream 成功响应和错误响应尽量原样透传。
5. gateway 自己产生的错误语义清晰、状态码稳定。
6. SSE streaming 行为与直连 upstream 一致。
7. 单 upstream 后面的副本池、健康和扩缩容由 Ray Serve/vLLM 服务端负责。

---

## 7. 风险与回滚

主要风险：

1. 客户端或测试此前依赖本地 Pydantic response object，而 proxy 路径现在返回原始 `Response`。
2. 上游返回非 JSON 错误时，客户端需要按上游 content type 自行处理。
3. 流式资源关闭需要靠测试覆盖，避免连接泄漏。

回滚策略：

1. 保留本地 `vllm` backend 配置。
2. 按模型粒度切换 backend。
3. 异常时将模型 backend 切回已验证的 `vllm` 路径。
