# vLLM Native Serving Adapter Refactor Summary

Date: 2026-05-31

## 背景

这次重构基于 `VLLM_NATIVE_SERVING_ADAPTER_REFACTOR_PLAN.md` 开始推进，目标是保留当前已经验证过的运行时形态：

```text
FastAPI gateway
  -> RuntimeDispatcher
  -> Ray Serve replica
  -> VLLMBackend
  -> local vLLM engine
```

同时把 OpenAI/vLLM 协议兼容性相关逻辑从 `src/infer_nexus/backends/vllm.py` 中逐步拆出，放到任务专属 adapter 中。这样 `VLLMBackend` 后续可以更接近“engine lifecycle + task dispatch”，而不是继续承担聊天协议、流式 delta 重建、多模态转换、embedding/rerank 响应塑形等所有细节。

本轮环境不能启动真实 vLLM，因此验证范围限定为语法检查和 mock 单测。

## 本轮完成的核心改动

### 1. 引入 `compat_mode`

新增兼容模式枚举：

```python
CompatibilityMode.VLLM_NATIVE = "vllm_native"
CompatibilityMode.LOCAL_BEST_EFFORT = "local_best_effort"
CompatibilityMode.STRICT_OPENAI = "strict_openai"
```

语义如下：

- `vllm_native`: 严格走 replica-local vLLM native serving adapter。adapter 缺失或调用失败时直接报错，不静默降级到本地 fallback。
- `local_best_effort`: 保留现有本地 vLLM API 路径，例如 `AsyncLLMEngine.generate`、`LLM.chat`、`LLM.embed`、`LLM.score`。
- `strict_openai`: 保留为旧配置兼容别名，目前只作为 chat 配置的兼容入口。

涉及文件：

- `src/infer_nexus/core/enums.py`
- `src/infer_nexus/catalog/models.py`
- `src/infer_nexus/core/schemas.py`
- `src/infer_nexus/runtime/dispatcher.py`
- `src/infer_nexus/runtime/serve_app.py`
- `src/infer_nexus/api/openai_routes.py`

### 2. 新增 `vllm_native` 模块

新增目录：

```text
src/infer_nexus/backends/vllm_native/
  __init__.py
  common.py
  chat.py
  embedding.py
```

当前职责：

- `chat.py`
  - 定义 `OpenAIChatServingAdapter` contract
  - 提供 `DynamicVLLMOpenAIChatServingAdapter`
  - 包装 vLLM OpenAI chat serving 对象
  - 非流式返回 dict payload
  - 流式 chunks 原样透出 bytes/string/dict

- `common.py`
  - 放置 vLLM native serving 共享结构和兼容代理
  - 包含 `ResolvedOpenAIServingImports`
  - 包含 `OpenAIServingEngineClientCompatProxy`

- `embedding.py`
  - 定义 `OpenAIEmbeddingServingAdapter` contract
  - 真实 vLLM 0.18 embedding serving 构造还未完成，需要目标环境验证

### 3. 收敛 `VLLMBackend` 中 native chat 行为

`VLLMBackend` 当前行为：

- `vllm_native` chat:
  - 构造 OpenAI-style request payload
  - 只重写 `model` 为 `served_model_name`
  - 合并 `extra_body`
  - 保留 `messages`、工具、reasoning、stream options、多模态 block 等 schema 字段
  - 调用 native chat serving adapter
  - adapter 缺失时报 `BackendConfigurationError`
  - adapter 调用失败直接抛出，不 fallback

- `local_best_effort` chat:
  - 保留现有本地路径
  - async engine 可用时使用 `AsyncLLMEngine`
  - 本地流式输出仍通过 `chat_delta` 事件和 executor SSE mapper 转换
  - sync `LLM.chat` 作为 fallback

### 4. 增加 native embedding contract 和严格路径

`EmbeddingRequest` 现在允许 vLLM 扩展字段，并显式支持：

```python
dimensions: int | None
```

`vllm_native` embedding 当前具备：

- adapter 注入 contract
- payload 构造逻辑
- `model` 重写为 `served_model_name`
- 保留 `input`、`encoding_format`、`dimensions`、`user`
- adapter 缺失时报错，不返回本地 stub

尚未完成：

- 自动构造真实 vLLM 0.18 embeddings native serving object
- 在 Linux/GPU + Ray 2.48.0 + vLLM 0.18.x 上对齐 `vllm serve /v1/embeddings`

### 5. 明确 rerank 暂不支持 native mode

本轮按方案保留 rerank 为 `local_best_effort`：

- 本地 rerank 继续使用 `LLM.score`
- `compat_mode: vllm_native` + rerank 会被拒绝
- 原因是 vLLM rerank/score serving 不是 OpenAI 官方协议，后续需要单独设计 protocol profile

### 6. Runtime streaming passthrough 更新

`RuntimeExecutor` 现在会根据 `compat_mode` 选择 stream response 处理方式：

- `vllm_native` 或旧别名 `strict_openai`:
  - 使用 passthrough stream response
  - 不把 native serving chunks 当成本地 `chat_delta` 事件重建

- `local_best_effort`:
  - 继续走本地 mapped chat stream response

### 7. 架构文档已更新

`docs/ARCHITECTURE.md` 已同步当前设计：

- 新增 `vllm_native` 模块关系
- 说明 `compat_mode` 语义
- 区分 `vllm_native` 和 `local_best_effort` 请求流
- 标注 native embedding 尚待真实 vLLM 环境验证
- 标注 rerank 保持 local-best-effort

## 已新增或调整的测试

新增/调整 mock 单测覆盖：

- native chat payload passthrough
- native chat stream chunk passthrough
- native chat adapter 缺失时失败
- native chat adapter 调用失败时不 fallback
- native embedding payload passthrough
- native embedding adapter 缺失时失败
- rerank 拒绝 `vllm_native`
- native serving dynamic import 初始化路径

主要测试文件：

- `tests/test_runtime.py`

## 本地验证结果

通过：

```bash
.venv/bin/python -m compileall src tests
```

通过：

```bash
.venv/bin/python -m pytest tests/test_runtime.py -k "vllm_backend_chat_completion_passthroughs_openai_serving_payload or vllm_backend_chat_completion_stream_passthroughs_openai_serving_chunks or vllm_native_chat_missing_adapter_fails_without_local_fallback or vllm_native_chat_invocation_failure_does_not_fallback or vllm_native_embedding_passthroughs_openai_serving_payload or vllm_native_embedding_missing_adapter_fails_without_local_fallback or rejects_vllm_native_rerank_runtime_spec or openai_serving_adapter_init"
```

结果：

```text
8 passed
```

未通过的广跑情况：

```bash
.venv/bin/python -m pytest tests/test_runtime.py tests/test_dispatcher.py tests/test_proxy_streaming.py tests/test_api.py
```

结果中存在大量失败，主要原因是当前 `config/models.yaml` 已经切换成真实模型配置，而部分旧测试仍期待旧的测试模型名，例如：

- `qwen3-chat`
- `bge-embedding`
- `bge-rerank`

这些失败与本次 native adapter contract 单测不是同一问题面，需要后续单独整理测试 fixture 或测试 catalog。

环境限制：

- `python -m compileall src tests` 使用默认 `python` 失败，因为当前 Poetry venv 指向的 Python 动态库缺失。
- `uv run ...` 失败，因为当前 `pyproject.toml` / `uv.lock` 存在解析问题。
- 已改用 `.venv/bin/python` 完成本轮语法检查和 mock 单测。

## 当前行为矩阵

| Task | `vllm_native` | `local_best_effort` |
| --- | --- | --- |
| chat | 已有 native adapter contract 和动态构造路径；mock 验证通过；真实 vLLM 0.18 仍需目标环境验证 | 保留现有 async/sync local engine 路径 |
| embedding | 已有 adapter contract 和 payload passthrough；真实 native serving 构造待完成 | 保留现有 `LLM.embed` 路径 |
| rerank | 明确拒绝 | 保留现有 `LLM.score` 路径 |

## 尚未完成

1. 把所有 local-best-effort chat / embedding / rerank 逻辑进一步拆到 `vllm_local/*`。

   这条指的是：当前 `src/infer_nexus/backends/vllm.py` 里还保留了大量本地 fallback 逻辑，例如 chat 的 sampling params 构造、`AsyncLLMEngine.generate` 调用、sync `LLM.chat` fallback、本地 chat delta 重建、`LLM.embed` 结果转换、`LLM.score` rerank 结果转换等。

   按重构目标，`vllm.py` 最终应该主要负责 engine 生命周期、runtime spec 校验、task dispatch 和 adapter 选择；这些 local-best-effort 协议处理应继续拆到类似下面的结构中：

   ```text
   src/infer_nexus/backends/vllm_local/
     chat.py
     embedding.py
     rerank.py
   ```

   这样 `vllm_native` 和 `local_best_effort` 两条路径的边界会更清楚，也能降低 `vllm.py` 继续膨胀的风险。

2. 自动构造真实 vLLM 0.18 native embeddings serving adapter。

   这条指的是：当前 embedding 的 native 模式只完成了 adapter contract 和严格调度路径。也就是说，`OpenAIEmbeddingServingAdapter` 已经存在，`VLLMBackend.embedding()` 在 `compat_mode=vllm_native` 时会构造 OpenAI-style embedding payload 并调用注入的 adapter；如果 adapter 缺失，会明确报错，不会 fallback 到本地 stub。

   但目前还没有真正实现“如何从 vLLM 0.18 内部类构造 embeddings serving 对象”。chat 已经有动态 import + serving object 包装路径，embedding 还需要在真实 vLLM 0.18 环境中确认 `/v1/embeddings` 使用的 request class、serving class、构造参数和 engine client 接法，然后补出类似 `DynamicVLLMOpenAIEmbeddingServingAdapter` 的实现。

3. 在目标环境中验证 native chat adapter 与 `vllm serve` 行为一致。
4. 在目标环境中验证 native embedding adapter 与 `vllm serve /v1/embeddings` 行为一致。
5. 整理旧测试 fixture，使广跑测试不依赖已经过期的 `config/models.yaml` 模型名。
6. 清理或处理当前 `uv` 配置/lockfile 解析问题。
7. 决定是否迁移旧配置中的 `strict_openai` 到 `vllm_native`，以及是否保留长期兼容。

## 建议下一步

优先顺序建议：

1. 先整理测试 catalog fixture，让 runtime / dispatcher / API 广跑恢复稳定。
2. 继续 Phase 2，把 local-best-effort 逻辑拆到 `src/infer_nexus/backends/vllm_local/`。
3. 在目标 Linux/GPU 环境检查 vLLM 0.18 embeddings serving internals，补齐 `VLLMNativeEmbeddingServingAdapter`。
4. 在目标环境对比 `infer-nexus vllm_native` 与 `vllm serve` 的 chat/embedding payload 和 stream 行为。
5. 最后再考虑 rerank native protocol profile，不要混入当前 OpenAI-compatible refactor。
