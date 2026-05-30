# infer-nexus / MinerU 联调排障报告

日期：2026-05-20  
分支：`refactor-per-model-serve-apps`

## 背景

本次排障的目标是让 `infer-nexus` 作为 OpenAI 兼容服务，稳定承接 MinerU 的 HTTP 推理请求，尤其是：

- `GET /v1/models`
- `POST /v1/chat/completions`

联调对象主要是 `mineru` 模型，对应目录中的：

- 模型名：`MinerU2.5-2509-1.2B`
- alias：`mineru`

本次问题并不是单一故障，而是几个不同层次的问题叠加出现，具体包括：

- Serve 应用组织方式带来的复杂度和启动竞态
- 服务端错误分类不清，导致真实异常被伪装成 `501`
- 远端环境代码不同步，导致 gateway 启动失败
- vLLM 版本兼容问题，导致 `SamplingParams` 参数不被接受

## 最终结论

今天最终定位并解决的核心兼容性问题是：

`mineru` 请求中携带的 `vllm_xargs.no_repeat_ngram_size` 被透传到了 `vllm.SamplingParams(...)`，但当前运行环境中的 vLLM 版本并不支持这个参数，最终在 Serve replica 内部抛出：

```text
TypeError: Unexpected keyword argument 'no_repeat_ngram_size'
```

在这之前，还顺带修复了几类会干扰定位的问题，包括：

- 单应用聚合根 `infer-nexus-root` 带来的额外复杂度
- 启动完成与实际可推理之间的时序问题
- 把真实执行异常错误地包装成 `501 Not Implemented`

## 问题时间线

### 1. 初始现象：`/v1/models` 正常，`/v1/chat/completions` 返回 `501`

最早的现象是：

- `GET /v1/models` 返回 `200`
- MinerU 在并发调用 `POST /v1/chat/completions` 时返回 `501`

这说明：

- gateway 进程已经启动
- 模型目录已经加载
- 但推理执行链路没有真正打通，或者真实执行异常被错误包装

### 2. 识别出 Serve 拓扑复杂度问题

原有实现使用了：

- 一个 Serve application：`infer-nexus`
- 多个模型 deployment
- 一个额外的 root deployment：`infer-nexus-root`

这类结构本身不是错误，但对本项目来说有两个问题：

- 它增加了额外的 deployment / replica 和启动噪音
- 让“应用名”和“模型 deployment 名”之间的关系更绕，不利于排障

因此后续将架构调整为：

- 每个模型一个独立 Serve app
- 每个 app 直接由该模型 deployment 充当 ingress

### 3. 启动时序问题暴露

早期的 `501` 一度怀疑是“服务刚起来，但 deployment 尚未 ready”。

这是合理怀疑，因为：

- `/v1/models` 是 gateway 自身返回
- `/v1/chat/completions` 则需要真正调用 Serve deployment
- 如果 gateway 比 deployment 更早暴露，客户端可能在应用层看到“端口可用，但模型未就绪”

为此做了启动时序修复：

- `run_serve_runtime.py` 在非阻塞模式下等待 Serve app 和 deployment 全部 `RUNNING/HEALTHY`
- `start_minimal.sh` 在启动 gateway 前，等待 runtime 真正 ready

### 4. 错误分类误导排障

后续发现一个更大的问题：

- 所有 Serve 执行阶段的异常，都会被包装成 `RuntimeNotConnectedError`
- API 层统一把它映射成 `501`

这会造成误导：

- `501` 本来意味着“未实现/未连通”
- 但实际很多情况是“已经连通，只是远端执行报错”

这导致最初看到 `501` 时，很难判断问题是在：

- handle 解析阶段
- app/deployment 路由阶段
- 还是 replica 内部执行阶段

因此做了错误分类修复：

- `RuntimeNotConnectedError` 保留给“连接/句柄不可用”
- 新增 `RuntimeExecutionError` 表示“远端执行失败”
- API 层将其映射为 `500 Internal Server Error`
- 同时打印完整 traceback

这一步是后续精准定位真实根因的关键。

### 5. 中间过程出现过一次 `Connection refused`

在某一轮测试中，MinerU 在 `GET /v1/models` 阶段直接报：

```text
Connection refused
```

这个问题与后面的 vLLM 问题不同，它是一个单独的部署问题：

- gateway 根本没起来
- 或者进程启动后立刻退出

随后拿到 gateway 启动日志，发现是因为代码不同步：

```text
ImportError: cannot import name 'RuntimeExecutionError' from 'infer_nexus.core.errors'
```

原因是：

- `executor.py` 已经引用了新加的 `RuntimeExecutionError`
- 但远端环境里的 `core/errors.py` 还是旧版本

这是典型的“半套代码发布”问题。

### 6. 真正根因被暴露：vLLM 参数兼容失败

当错误分类理顺、服务成功启动后，真正的异常 finally 暴露出来：

```text
TypeError: Unexpected keyword argument 'no_repeat_ngram_size'
```

调用链为：

1. MinerU 构造 chat completion 请求
2. 请求中带入 `vllm_xargs.no_repeat_ngram_size`
3. `infer-nexus` 在 [src/infer_nexus/backends/vllm.py](./src/infer_nexus/backends/vllm.py) 中把它放进 `sampling_params`
4. Serve replica 内部调用：

```python
SamplingParams(**sampling_params)
```

5. 当前运行环境中的 vLLM 版本不支持该参数
6. replica 抛出 `TypeError`
7. gateway 接收到 RayTaskError，并返回 `500`

### 7. 为什么单图测试能成功，MinerU PDF 流程会失败

这是本次联调中的一个关键迷惑点。

单独的 OpenAI client 测试脚本可以成功，说明：

- gateway 路由没问题
- `mineru` alias 解析没问题
- 基础 multimodal message 格式没问题

但 MinerU PDF 流程仍然失败，原因在于两者请求内容并不完全一致：

- 单图脚本是一个简化请求
- MinerU `doc_analyze(...)` 会带入自己的额外参数
- 其中包含 `vllm_xargs.no_repeat_ngram_size`

也就是说，问题不是“图像推理整体不可用”，而是“MinerU 这条调用链特有的兼容参数触发了当前 vLLM 版本不支持的路径”。

## 本次代码修改说明

### 一、Serve 应用结构调整

目的：

- 去掉 `infer-nexus-root`
- 每个模型一个独立 Serve app
- 降低拓扑复杂度和排障复杂度

影响文件包括：

- [src/infer_nexus/runtime/serve_app.py](./src/infer_nexus/runtime/serve_app.py)
- [src/infer_nexus/runtime/types.py](./src/infer_nexus/runtime/types.py)
- [src/infer_nexus/runtime/dispatcher.py](./src/infer_nexus/runtime/dispatcher.py)
- [src/infer_nexus/runtime/handles.py](./src/infer_nexus/runtime/handles.py)
- [src/infer_nexus/runtime/executor.py](./src/infer_nexus/runtime/executor.py)
- [src/infer_nexus/main.py](./src/infer_nexus/main.py)
- [scripts/run_serve_runtime.py](./scripts/run_serve_runtime.py)

核心变化：

- `app_name` 从固定单值变成按模型解析
- `ServeDeploymentHandleResolver` 不再持有全局 app name
- `run_serve_runtime.py` 改为逐模型 `serve.run(...)`

### 二、启动 ready 等待机制增强

目的：

- 避免 runtime 尚未 ready 时 gateway 先暴露

相关文件：

- [scripts/run_serve_runtime.py](./scripts/run_serve_runtime.py)
- [scripts/start_minimal.sh](./scripts/start_minimal.sh)

核心变化：

- 部署完成后主动轮询 `serve.status()`
- 仅当目标 app 及其 deployment 全部进入 `RUNNING/HEALTHY` 才视为 ready

### 三、错误分类重构

目的：

- 将“未连通”和“执行失败”分开

相关文件：

- [src/infer_nexus/core/errors.py](./src/infer_nexus/core/errors.py)
- [src/infer_nexus/runtime/executor.py](./src/infer_nexus/runtime/executor.py)
- [src/infer_nexus/api/openai_routes.py](./src/infer_nexus/api/openai_routes.py)

核心变化：

- 新增 `RuntimeExecutionError`
- `_invoke_handle()` 中远端执行异常不再包装成 `RuntimeNotConnectedError`
- API 层对 `RuntimeExecutionError` 返回 `500`
- 加入 `logger.exception(...)` 打印 traceback

### 四、vLLM `SamplingParams` 兼容逻辑增强

这是本次最终解决真实根因的关键改动。

相关文件：

- [src/infer_nexus/backends/vllm.py](./src/infer_nexus/backends/vllm.py)
- [tests/test_runtime.py](./tests/test_runtime.py)

#### 第一层兼容：按签名过滤

原理：

- 使用 `inspect.signature(SamplingParams.__init__)`
- 过滤掉不在当前签名中的字段

这本来可以解决很多版本差异，但本次发现还不够，因为：

- 某些 vLLM 版本对外暴露的签名和实际支持参数并不完全一致

#### 第二层兼容：按异常动态剔除

最终采用的方式是：

1. 先做一轮签名过滤
2. 尝试构造 `SamplingParams(**filtered_sampling_params)`
3. 如果抛出：

```text
TypeError: Unexpected keyword argument 'xxx'
```

则：

- 从参数字典中删掉 `xxx`
- 重新尝试构造

直到：

- 构造成功
- 或遇到不是“未知参数”的 `TypeError`

这样可以兜住以下情况：

- 新版参数在旧版 vLLM 不存在
- 签名 introspection 结果不可靠
- 第三方库暴露签名与真实校验逻辑不一致

## 问题成因分析

本次故障的根因可以拆成四层。

### 1. 架构层：Serve 应用拓扑偏复杂

单 app + root deployment 并不是错，但对当前项目而言：

- 模型本来就是天然独立单元
- gateway 又已经承担统一入口

因此聚合 root deployment 的收益较低，复杂度偏高。

### 2. 生命周期层：服务 ready 与端口可用不是同一回事

这是很多服务联调里常见的问题。

一个 HTTP 端口能响应，不等于：

- 所有依赖都 ready
- 后端 actor 已经健康
- deployment handle 已可用
- 模型引擎已完成初始化

MinerU 这类会发起并发请求的客户端，尤其容易放大这种时序问题。

### 3. 观测层：错误分类不准会直接拖慢排障

最初把远端执行错误都映射成 `501`，导致判断方向被带偏：

- 一开始怀疑是“未实现 / 未连接”
- 实际问题在 replica 内部执行

一旦错误语义不准，排障成本会直线上升。

### 4. 兼容层：对第三方库参数能力假设过强

本次最终根因本质上是：

- 请求侧带入了一个“vLLM 兼容参数”
- 服务侧默认相信运行环境的 vLLM 一定支持它

这类假设在多版本环境里不成立。

特别是：

- 模型服务常跨机器部署
- vLLM / transformers / CUDA / Ray 版本组合可能不同
- 第三方库的公开签名与实际 runtime 行为也可能不完全一致

## 解决方法总结

### 已完成修复

1. 将 Serve 架构改为每模型一个 app，去掉 `infer-nexus-root`
2. 给 runtime 启动加入健康等待
3. 把远端执行失败从 `501` 改为 `500`
4. 增加服务端完整 traceback 日志
5. 为 vLLM `SamplingParams` 增加双层兼容处理：
   - 按签名过滤
   - 按运行时报错逐步剔除未知字段

### 当前行为变化

修复后，系统行为更合理：

- 连接/句柄问题：返回 `501`
- 远端执行问题：返回 `500`
- 对旧版 vLLM 不支持的采样参数：自动降级，而不是直接崩溃

## 为什么这次修复有效

修复的关键不只是“把一个字段删掉”，而是改成了一个更稳的兼容策略。

如果只是硬编码删除 `no_repeat_ngram_size`：

- 这次问题会消失
- 但下次可能换成 `min_p`、`top_k`、`prompt_logprobs`、`skip_special_tokens` 或其他字段再次出现

现在采用的策略更通用：

- 先保留尽可能多的参数能力
- 再根据当前 vLLM 运行时实际能力进行降级

这意味着：

- 新版环境不会损失能力
- 旧版环境不会因为少数字段不支持而整体不可用

## 未来如何避免类似兼容性问题

### 1. 为第三方运行时能力建立“显式适配层”

建议不要把来自上游客户端的扩展字段直接透传给底层推理库。

更稳妥的做法是：

- 在 `infer-nexus` 内定义自己的采样参数白名单
- 每个 backend 明确声明“自己支持哪些参数”
- 多余参数统一做降级、忽略或告警

建议方向：

- 在 backend 层维护 `supported_sampling_fields`
- 对不支持字段输出 `warning`，而不是让底层库直接抛异常

### 2. 增加版本矩阵兼容测试

建议至少覆盖这些组合：

- Ray Serve 不同版本
- vLLM 不同版本
- MinerU 请求中常见 extra fields 组合

重点测试项：

- `top_k`
- `min_p`
- `skip_special_tokens`
- `prompt_logprobs`
- `no_repeat_ngram_size`
- 其他 `vllm_xargs`

### 3. 将“运行时能力探测”前置

可以在服务启动时主动检测：

- `SamplingParams` 支持哪些字段
- `LLM.__init__` 支持哪些字段
- 当前引擎支持哪些 task mode

然后把结果缓存到 runtime context 或日志中。

这样启动时就能看到：

- 当前环境能力边界
- 哪些字段会被自动忽略

### 4. 发布时强制做完整代码一致性检查

本次中途出现过“半套代码发布”的问题，说明部署流程也需要约束。

建议：

- 部署前执行 `git rev-parse HEAD` 并记录版本
- gateway / runtime / test client 使用同一 commit
- 启动时打印核心模块版本或 commit hash

### 5. 将 ready 检查从“进程启动”提升为“端到端探活”

单纯探测端口或 `/v1/models` 不够。

更好的健康检查应分层：

- L1：进程是否存活
- L2：gateway 是否可响应
- L3：Serve app / deployment 是否健康
- L4：是否能完成一次真实的最小推理请求

对 MinerU 这种强依赖多模态 chat 的客户端，L4 才是最有价值的准入检查。

### 6. 建议保留当前错误分类设计

后续不要再把所有运行时异常压成同一个错误码。

至少要区分：

- `runtime_not_connected`
- `runtime_execution_failed`
- `backend_configuration_error`
- `unsupported_parameter`
- `model_artifact_missing`

这样问题才会在第一时间显性化。

## 建议的后续工作

### 短期

1. 在远端环境同步并验证当前分支全部改动
2. 用 MinerU PDF 流程再做一次完整回归
3. 补充一个最小多页 PDF 的稳定性测试

### 中期

1. 为采样参数兼容写成独立工具函数或适配器
2. 增加“真实 vLLM 版本差异”的单元测试或集成测试
3. 为 gateway 增加更直接的 `/readyz` 深度检查

### 长期

1. 建立 backend capability registry
2. 把客户端请求能力与后端实际能力做显式协商
3. 将部署版本、依赖版本、模型版本统一纳入可观测性输出

## 结语

今天的问题表面上看像是：

- Serve 路由错了
- gateway 不稳定
- MinerU 请求有问题

但最后证明，真正的核心故障是一个典型的运行时兼容问题：

- 上游请求带了扩展参数
- 下游推理库版本不支持
- 中间层一开始又缺乏足够准确的错误分类和日志

这次修复的价值不只是“把当前问题修好”，更重要的是把系统从：

- 难以定位
- 容易误导
- 对第三方版本差异脆弱

调整成了：

- 更容易定位
- 错误语义更准确
- 对 vLLM 版本差异更有弹性

如果后续继续沿着“能力探测 + 适配层 + 分层健康检查 + 清晰错误语义”这条方向演进，类似问题会明显减少。
