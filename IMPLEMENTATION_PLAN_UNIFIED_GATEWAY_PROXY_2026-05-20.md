# infer-nexus 统一入口 + 自托管 vLLM Proxy 模式实施清单

日期：2026-05-20  
适用分支：`refactor-per-model-serve-apps`  
目标读者：开发、测试、运维

---

## 1. 背景与目标

当前项目希望达到的架构目标是：

1. 对外只有一个统一入口（OpenAI 兼容 `/v1/*`）。
2. 所有模型通过请求体 `model` 字段路由。
3. 模型实例仍由团队自托管（不是第三方托管）。
4. 网关负责治理（鉴权、限流、路由、观测、错误边界），尽量不重写 vLLM 协议语义。

本清单对应的改造方向是：

- 将生产主路径收敛到 `vllm_openai_proxy`。
- 保留本地 `vllm` backend 作为 fallback/实验路径。
- 减少因本地参数拼装（如 `SamplingParams`）导致的版本兼容问题。

---

## 2. 总体架构（目标态）

```text
Client
  -> infer-nexus gateway (/v1/*)
      -> model routing by request.model
          -> upstream vLLM server instance pool (self-hosted)
```

关键原则：

1. 网关只做最小必要改写（通常仅 `model` 字段）。
2. 请求体其余字段尽量原样透传。
3. 上游 HTTP 状态码和错误体尽量原样返回。
4. 流式场景做 SSE 字节透传，不做语义重组。

---

## 3. 范围与非范围

### 3.1 本次范围

1. `chat/completions` 路径优先完善。
2. 多 upstream 实例池能力（同模型多个上游）。
3. 启动探测 + 周期健康探测 + 故障摘除。
4. 观测字段与指标补齐。
5. 配置、文档、测试体系同步更新。

### 3.2 暂不纳入

1. 动态模型注册。
2. 跨地域智能路由。
3. 全量新 API（如 responses API）深度支持。

---

## 4. 文件级改动清单（详细）

以下路径均为仓库绝对路径，便于直接定位实施。

### 4.1 配置层

#### A. `/Users/zhoushujian/Projects/GitHub/infer-nexus/config/models.yaml`

改动项：

1. 生产模型默认切换为 `backend: vllm_openai_proxy`。
2. 为每个模型补全 `proxy_config`：
   - `upstream_base_url`
   - `upstream_model_name`
   - `timeout.*`
   - `retry.*`
   - `streaming.*`
   - `headers_policy.*`
3. 新增多 upstream 配置结构（建议）：
   - `proxy_config.upstreams: [{name, base_url, weight, enabled, timeout_override, ...}]`
4. 保留 fallback 配置模板（本地 `vllm`），用于紧急回滚。

验收点：

1. `mineru` 能通过 `model=mineru` 走 proxy。
2. `/v1/models` 返回模型标识不变。
3. 切换不需要客户端改 base_url 或 path。

---

### 4.2 Schema 与配置校验层

#### B. `/Users/zhoushujian/Projects/GitHub/infer-nexus/src/infer_nexus/catalog/models.py`

改动项：

1. 扩展 `ProxyConfig` 结构支持实例池。
2. 兼容单 upstream 老字段（避免一次性破坏现有配置）。
3. 增加校验规则：
   - `backend == vllm_openai_proxy` 时必须有可用 upstream。
   - timeout/retry 参数取值范围合法。
   - 至少一个 upstream `enabled=true`。
4. 为后续路由策略预留字段：
   - `routing_policy`（round_robin / least_failures）
   - `circuit_breaker`（阈值、恢复窗口）

验收点：

1. 非法配置在启动前失败并给出明确错误信息。
2. 老配置仍可读取（带 deprecation 提示）。

---

### 4.3 Runtime 执行层（核心）

#### C. `/Users/zhoushujian/Projects/GitHub/infer-nexus/src/infer_nexus/runtime/executor.py`

改动项（核心）：

1. 保持 `_proxy_payload` 最小改写策略：
   - 仅改写 `model`（映射到 `upstream_model_name`）。
   - 其余字段保持原样。
2. 引入多 upstream 路由：
   - 读取实例池配置
   - 按策略选择实例
   - 失败重试时可切换实例
3. 健康与熔断：
   - 实例连续失败计数
   - 达阈值临时摘除
   - 恢复窗口后探测恢复
4. 保持上游错误原样透传：
   - 4xx/5xx status 和 body 保留
5. SSE 流保持字节透传：
   - 不解析事件，不重组 chunk
6. 结构化日志增强：
   - `request_id`
   - `public_model`
   - `upstream_model`
   - `upstream_instance`
   - `status_code`
   - `latency_ms`

验收点：

1. 同一请求经网关与直连上游响应语义一致。
2. 单实例故障时请求可自动路由到健康实例。
3. 网关不会再触发本地 `SamplingParams` 参数兼容异常（proxy 路径）。

---

### 4.4 Runtime 分发层

#### D. `/Users/zhoushujian/Projects/GitHub/infer-nexus/src/infer_nexus/runtime/dispatcher.py`

改动项：

1. `resolve_target` 中补全 proxy 运行时上下文：
   - upstream 实例池
   - 路由策略
   - 观测标签
2. 显式声明 backend 分流优先级：
   - 生产默认 proxy
   - 本地 `vllm` 仅 fallback

验收点：

1. 路由行为可预测，不出现隐式 fallback。

---

### 4.5 API 路由与错误语义层

#### E. `/Users/zhoushujian/Projects/GitHub/infer-nexus/src/infer_nexus/api/openai_routes.py`

改动项：

1. 保持网关错误与上游错误分层：
   - 网关阶段失败：统一网关错误对象
   - 上游阶段失败：尽量原样透传
2. 响应头增加追踪标识（如 `X-Infer-Nexus-Request-ID`），不改 payload。
3. 对 proxy 场景补充日志上下文字段。

验收点：

1. 用户能从响应和日志快速区分“网关错误”与“上游错误”。

---

### 4.6 错误类型层

#### F. `/Users/zhoushujian/Projects/GitHub/infer-nexus/src/infer_nexus/core/errors.py`

改动项：

1. 增补 proxy 相关错误类型（如全部 upstream 不可用）。
2. 维持现有边界，不把执行错误重新混到 `501`。
3. 规范错误 `code`，便于平台侧聚合。

验收点：

1. 错误码稳定且可用于告警规则。

---

### 4.7 启动脚本与运行模式

#### G. `/Users/zhoushujian/Projects/GitHub/infer-nexus/scripts/start_minimal.sh`

改动项：

1. 增加 `proxy-only` 启动模式（仅 gateway，不强制拉起本地 Serve runtime）。
2. 启动日志明确打印当前模式：
   - `local-runtime`
   - `proxy-only`
3. 状态文件写入模式信息，便于排障。

验收点：

1. 一条命令即可启动“统一入口 + 上游 proxy”模式。

#### H. `/Users/zhoushujian/Projects/GitHub/infer-nexus/scripts/run_gateway.py`

改动项：

1. 启动时打印模型路由摘要（脱敏）。
2. 打印每个模型上游连通性预检结果（可选开关）。

验收点：

1. 启动后可快速确认配置是否生效。

#### I. `/Users/zhoushujian/Projects/GitHub/infer-nexus/scripts/run_serve_runtime.py`

改动项：

1. 明确标注仅用于本地 runtime backend。
2. 文档和脚本帮助中提示“生产推荐 proxy-only + 自托管上游”。

验收点：

1. 避免误把 runtime 模式当成生产默认。

---

### 4.8 文档层

#### J. `/Users/zhoushujian/Projects/GitHub/infer-nexus/README.md`

改动项：

1. 更新推荐架构说明：单入口网关 + 自托管上游实例。
2. 补充模型接入步骤：
   - 新增模型配置
   - 上游探测
   - 灰度放量
3. 补充回滚流程。

#### K. `/Users/zhoushujian/Projects/GitHub/infer-nexus/MINIMAL_STARTUP.md`

改动项：

1. 增加 `proxy-only` 启动流程。
2. 增加 `curl` 联调范例（普通/流式）。
3. 增加常见问题排障段落。

#### L. `/Users/zhoushujian/Projects/GitHub/infer-nexus/ARCHITECTURE.md`

改动项：

1. 更新拓扑图与职责划分。
2. 强调“不在网关层重实现 vLLM 采样语义”原则。

---

### 4.9 测试层

#### M. `/Users/zhoushujian/Projects/GitHub/infer-nexus/tests/test_runtime.py`

新增/调整：

1. proxy payload 保真测试（除 `model` 外不变）。
2. 上游 4xx/5xx 原样透传测试。
3. 重试跨实例测试。

#### N. `/Users/zhoushujian/Projects/GitHub/infer-nexus/tests/test_openai_routes.py`

新增/调整：

1. 模型不存在、任务不匹配、配置错误的网关错误码稳定性。
2. proxy backend 下 `model` 路由行为一致性。

#### O. `/Users/zhoushujian/Projects/GitHub/infer-nexus/tests/test_proxy_streaming.py`（新建）

新增：

1. SSE `content-type` 保持测试。
2. 事件字节流不改写测试。
3. 上游流中断时的关闭行为测试。

#### P. `/Users/zhoushujian/Projects/GitHub/infer-nexus/tests/test_proxy_health_routing.py`（新建）

新增：

1. 健康探测状态机测试。
2. 摘除与恢复测试。
3. 轮询/策略选择测试。

---

## 5. 分阶段实施顺序（建议）

### Phase 1（核心链路）

1. 先改 `models.py` + `executor.py` + `dispatcher.py`。
2. 同步补 `test_runtime.py` / `test_proxy_streaming.py`。

完成标准：

1. `mineru` 能通过 proxy 正常服务。
2. 请求参数不再在网关层触发 vLLM 参数不兼容异常。

### Phase 2（可用性）

1. 加实例池、健康探测、摘除、恢复。
2. 补 `test_proxy_health_routing.py`。

完成标准：

1. 单 upstream 故障不影响整体可用。

### Phase 3（运维与文档）

1. 启动脚本支持 `proxy-only`。
2. 更新 README / ARCHITECTURE / MINIMAL_STARTUP。

完成标准：

1. 新同学可按文档完成部署、联调、排障。

### Phase 4（灰度上线）

1. `mineru` 先 5% -> 25% -> 100% 灰度。
2. 对比“直连上游 vs 经网关”响应一致性。

完成标准：

1. 关键指标稳定，无新增 P1/P2 故障。

---

## 6. 验收标准（最终）

1. 对外仅单入口：`/v1/*`。  
2. 所有模型通过 `request.model` 路由。  
3. `mineru` 场景稳定，不再出现 `no_repeat_ngram_size` 类本地参数构造错误。  
4. 上游错误原样透传，网关错误语义清晰。  
5. SSE 流式透传行为与直连上游一致。  
6. 单实例故障具备自动摘除与恢复。  
7. 日志和指标可定位到具体 upstream 实例。  

---

## 7. 风险与回滚

主要风险：

1. 配置迁移时字段不兼容。
2. 多 upstream 路由引入边界状态 bug。
3. 流式场景资源关闭处理不完整。

回滚策略：

1. 保留本地 `vllm` backend 配置模板。
2. 模型粒度切换 backend（先 `mineru`，逐个切换）。
3. 灰度期异常立即回切至旧 backend。

---

## 8. 建议的交付工件

1. 代码 PR（按 Phase 分拆，避免超大 PR）。  
2. 配置迁移说明（旧字段 -> 新字段映射表）。  
3. 回归测试报告（普通/流式/故障注入）。  
4. 运行手册（启动、灰度、回滚、排障）。  

