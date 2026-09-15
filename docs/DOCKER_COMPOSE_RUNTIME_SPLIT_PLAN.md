# Docker Compose Runtime Split Implementation Plan

## 背景

当前本地启动主要依赖 `scripts/start_minimal.sh` 或
`scripts/start_minimal_ascend.sh` 串行完成以下工作：

1. 准备本地状态目录、日志目录和 PID 文件。
2. 可选执行依赖安装。
3. 检查或启动 Ray head。
4. 执行 `scripts/run_serve_runtime.py` 部署 Ray Serve 应用。
5. 执行 `scripts/run_gateway.py` 启动 FastAPI gateway。

这个方式适合早期 bring-up，但它把进程生命周期、健康检查、Ray
控制面、Serve 部署和 gateway 启动都压在一个 shell 脚本里。后续需要更清晰的
容器生命周期、自动重启、健康检查和组件间通信时，应该把这些职责交给 Docker
Compose。

本计划描述第一阶段的容器化重构方案。该阶段只拆分容器生命周期，不改变当前
`infer-nexus` 的核心运行时架构。

## 目标

第一阶段目标容器分布：

```text
gateway
  FastAPI / Uvicorn，一个容器内运行多个 gateway worker 进程

ray-head
  Ray 控制面

ray-worker
  Ray worker + Ray Serve replica + vLLM 推理进程

serve-deployer
  一次性执行 scripts/run_serve_runtime.py，提交 serve.run(...) 后退出
```

请求链路保持为：

```text
client
-> gateway container
-> Uvicorn worker process
-> FastAPI app
-> RuntimeDispatcher / RuntimeExecutor
-> Ray Serve deployment handle
-> Ray Serve replica inside ray-worker
-> replica-local vLLM runtime
```

Serve 部署链路为：

```text
serve-deployer container
-> scripts/run_serve_runtime.py
-> ray.init(address="ray-head:6379")
-> serve.start(proxy_location="Disabled")
-> serve.run(..., name=<per-model-app>, route_prefix=None)
-> wait until Serve applications are ready
-> exit 0
```

## 非目标

第一阶段不要做以下事情：

- 不引入 Nginx 或其他外部负载均衡器。
- 不拆成多个 gateway 容器。
- 不把 `gateway worker` 实现为单独的业务服务。
- 不把每个 Ray Serve deployment 实现为 Docker Compose service。
- 不替换 Ray Serve 的 deployment / replica 生命周期管理。
- 不实现 `runtime_worker_enabled` 对应的 runtime worker isolation。
- 不引入 Kubernetes、KubeRay 或跨集群调度。
- 不做动态模型注册。
- 不做多 backend 共存。

## 关键架构约束

### Compose 管容器，Ray 管 replica

Docker Compose 负责：

- 启动和重启容器。
- 容器依赖顺序。
- 容器健康检查。
- 网络名解析。
- 挂载配置、模型目录和缓存目录。

Ray Serve 继续负责：

- 每个模型一个 Serve application / deployment。
- deployment 生命周期。
- replica 调度。
- replica 健康和路由。
- Ray Serve 运行时指标。

不要把 Ray Serve deployment 直接建模为 Compose service。那会把当前
Ray Serve runtime 架构改成固定容器化 vLLM 服务架构，超出本阶段范围。

### gateway 容器内的 worker 语义

第一阶段的 `gateway` 容器中应运行：

```text
gateway container
-> Uvicorn parent process
   -> Uvicorn worker process 1
      -> FastAPI app instance
   -> Uvicorn worker process 2
      -> FastAPI app instance
   -> Uvicorn worker process N
      -> FastAPI app instance
```

不是：

```text
one FastAPI process
-> many gateway worker services
```

每个 Uvicorn worker 都会加载自己的 FastAPI app、`app.state`、registry、
dispatcher、executor、admission controller 和 Serve handle cache。

因此以下配置仍然是 process-local：

- `runtime.gateway_worker_max_inflight`
- `runtime.max_inflight_per_model`
- `runtime.max_streaming_inflight_per_model`
- `runtime.max_non_streaming_inflight_per_model`
- `runtime.max_queued_per_model`
- Serve deployment handle cache
- gateway Prometheus process metrics

这个语义与当前 `scripts/run_gateway.py --workers N` 保持一致。

## 目标文件变更

建议在实施分支中完成以下文件变更。

### 新增或重建 Dockerfile

建议文件：

```text
Dockerfile
```

第一阶段使用统一 runtime image，供 `gateway`、`ray-head`、`ray-worker` 和
`serve-deployer` 共用。这样能先减少镜像差异带来的环境问题。

推荐镜像能力：

- Python 3.11 或项目当前支持的 Python 版本。
- 安装项目基础依赖。
- 安装 `serve` extra。
- 安装 `vllm` extra。
- 可选安装 `artifacts` extra。
- 设置 `PYTHONPATH=/app/src`。
- 工作目录为 `/app`。

依赖安装建议基于当前项目 extras：

```bash
uv sync --preview-features extra-build-dependencies --extra serve --extra vllm
```

如果部署环境需要模型 artifact 下载能力：

```bash
uv sync --preview-features extra-build-dependencies --extra serve --extra vllm --extra artifacts
```

注意：

- 不要在镜像里复制或打包大模型权重。
- 模型目录应通过 volume 挂载到容器。
- 如果 CUDA / Ascend 运行环境需要专用 base image，实施时优先选择与现有
  vLLM / Ray / 驱动匹配的 runtime image，而不是从通用 Python 镜像硬装。

### 新增 docker-compose.yml

建议文件：

```text
docker-compose.yml
```

第一阶段服务：

```text
ray-head
ray-worker
serve-deployer
gateway
```

如果当前工作树已有旧 compose 文件，实施时应先确认用户是否要替换、保留或另存为
`docker-compose.runtime.yml`。不要无意覆盖用户未提交的 compose 变更。

### 新增 Compose 专用 settings

建议文件：

```text
config/settings.compose.yaml
```

目的：

- 避免直接把本地脚本启动配置和容器启动配置混在一起。
- 明确容器内路径。
- 明确 gateway 到 Ray head 的地址。

建议关键配置：

```yaml
service:
  host: 0.0.0.0
  port: 8000
  workers: 4

runtime:
  execution_mode: serve
  backend_init_mode: real
  ray_address: ray-head:6379

model_store:
  root_dir: /models
```

`service.workers` 数值应根据 CPU、请求并发和 Ray handle 稳定性保守设置。第一版可从
`1` 或 `2` 开始，生产压测后再提高。

### 补充 runtime.ray_address 配置

当前 `scripts/run_serve_runtime.py` 已经支持：

```bash
--ray-address
```

但 gateway 侧当前没有清晰的 Ray address 配置入口。Compose 模式下 gateway 容器
不能依赖本机 `auto`，应显式连接：

```text
ray-head:6379
```

建议新增配置字段：

```python
class RuntimeSettings(BaseModel):
    ray_address: str | None = None
```

建议语义：

- `None`：保持当前行为，适用于本地脚本或已经初始化的环境。
- 非空字符串：gateway 在 `execution_mode: serve` 时显式执行
  `ray.init(address=settings.runtime.ray_address, ignore_reinit_error=True, ...)`。

建议实现位置：

- `src/infer_nexus/core/config.py`
- `src/infer_nexus/main.py`
- 必要时扩展 `src/infer_nexus/runtime/handles.py`

推荐封装方式：

```text
main.lifespan()
-> load settings
-> if execution_mode == "serve" and runtime.ray_address is set:
     initialize Ray connection before creating ServeDeploymentHandleResolver
```

要避免：

- 每次请求里执行 `ray.init(...)`。
- 在非 `serve` 模式下强依赖 Ray。
- 在 test/stub 模式下破坏无 Ray 依赖的启动路径。

### 可选新增 Compose 入口脚本

如果 compose command 过长，可以新增轻量入口脚本：

```text
scripts/docker/run_ray_head.sh
scripts/docker/run_ray_worker.sh
scripts/docker/run_gateway.sh
scripts/docker/run_serve_deployer.sh
```

但第一阶段也可以直接在 `docker-compose.yml` 中写 command。优先保持简单。

## docker-compose.yml 设计草案

以下是实施时可参考的结构，不要求逐字照搬。

```yaml
services:
  ray-head:
    image: infer-nexus:runtime
    build:
      context: .
      dockerfile: Dockerfile
    command:
      - bash
      - -lc
      - >
        ray start --head
        --node-ip-address=0.0.0.0
        --port=6379
        --dashboard-host=0.0.0.0
        --disable-usage-stats
        --block
    ports:
      - "8265:8265"
    healthcheck:
      test: ["CMD-SHELL", "ray status --address=127.0.0.1:6379 >/dev/null 2>&1"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 20s
    restart: unless-stopped

  ray-worker:
    image: infer-nexus:runtime
    depends_on:
      ray-head:
        condition: service_healthy
    command:
      - bash
      - -lc
      - >
        ray start --address=ray-head:6379
        --disable-usage-stats
        --num-gpus=${RAY_WORKER_NUM_GPUS:-1}
        --block
    volumes:
      - ./config:/app/config:ro
      - ./models:/models
    healthcheck:
      test: ["CMD-SHELL", "ray status --address=ray-head:6379 >/dev/null 2>&1"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 30s
    restart: unless-stopped

  serve-deployer:
    image: infer-nexus:runtime
    depends_on:
      ray-head:
        condition: service_healthy
      ray-worker:
        condition: service_healthy
    command:
      - bash
      - -lc
      - >
        python scripts/run_serve_runtime.py
        --settings config/settings.compose.yaml
        --ray-address ray-head:6379
        --proxy-location Disabled
        --ready-timeout-seconds ${SERVE_READY_TIMEOUT_SECONDS:-900}
    volumes:
      - ./config:/app/config:ro
      - ./models:/models
    restart: "no"

  gateway:
    image: infer-nexus:runtime
    depends_on:
      serve-deployer:
        condition: service_completed_successfully
    command:
      - bash
      - -lc
      - >
        python scripts/run_gateway.py
        --settings config/settings.compose.yaml
        --host 0.0.0.0
        --workers ${GATEWAY_WORKERS:-4}
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config:ro
      - ./models:/models
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()\""]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 20s
    restart: unless-stopped
```

实施时需要根据实际 GPU / NPU 运行时补充设备映射。

CUDA 示例方向：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

或使用部署环境支持的 NVIDIA Container Toolkit 配置。

Ascend / NPU 示例方向：

```yaml
privileged: true
environment:
  ASCEND_RT_VISIBLE_DEVICES: "0,1,2,3"
  RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES: "1"
volumes:
  - /usr/local/Ascend:/usr/local/Ascend:ro
  - /dev:/dev
```

Ascend 具体设备和驱动挂载应以目标机器实际要求为准。

## 容器职责明细

### gateway

运行命令：

```bash
python scripts/run_gateway.py \
  --settings config/settings.compose.yaml \
  --host 0.0.0.0 \
  --workers ${GATEWAY_WORKERS:-4}
```

职责：

- 提供 OpenAI-compatible inference API。
- 提供 platform-native API。
- 提供 `/healthz`、`/readyz` 和 `/metrics`。
- 执行 gateway worker admission。
- 执行 per-model admission。
- 通过 Ray Serve deployment handle 调用后端模型。

注意：

- 第一阶段只有一个 gateway 容器。
- 容器内可以有多个 Uvicorn worker 进程。
- 不要同时使用 `--reload` 和多 worker。
- 后续如扩展到多个 gateway 容器，应在前面增加 Nginx / LB，并重新评估全局
  admission 语义。

### ray-head

运行命令：

```bash
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --disable-usage-stats \
  --block
```

职责：

- Ray control plane。
- Ray GCS。
- Ray dashboard。
- Ray Serve controller 所在控制面。

注意：

- 第一阶段建议尽量不在 `ray-head` 上承载 GPU/NPU 推理资源。
- 单机简化部署可以让 head 也暴露资源，但长期更建议推理资源放在
  `ray-worker`。

### ray-worker

CUDA 方向运行命令：

```bash
ray start --address=ray-head:6379 \
  --disable-usage-stats \
  --num-gpus=${RAY_WORKER_NUM_GPUS:-1} \
  --block
```

NPU 方向运行命令：

```bash
ray start --address=ray-head:6379 \
  --disable-usage-stats \
  --resources "{\"NPU\": ${RAY_WORKER_NUM_NPUS:-1}}" \
  --block
```

职责：

- 加入 Ray cluster。
- 暴露 CPU / GPU / NPU 资源。
- 承载 Ray Serve deployment replica。
- 在 replica 内初始化 `ModelRuntimeReplica`。
- 在 replica 内启动 vLLM backend。
- 执行真实模型推理。

注意：

- `config/models.yaml` 中的 `cpu_per_replica`、`gpu_per_replica`、
  `min_replicas`、`max_replicas` 仍然决定每个模型的资源需求和副本上下限。
- 不要在 Compose 层静态指定某个模型使用某张卡。
- 设备池边界属于 Ray runtime / container runtime；每个模型的资源需求属于
  model config。

### serve-deployer

运行命令：

```bash
python scripts/run_serve_runtime.py \
  --settings config/settings.compose.yaml \
  --ray-address ray-head:6379 \
  --proxy-location Disabled \
  --ready-timeout-seconds 900
```

职责：

- 读取 settings。
- 读取 model catalog。
- 为每个本地 `backend: vllm` 模型构建 Ray Serve binding。
- 对每个模型执行 `serve.run(...)`。
- 等待 Serve application / deployment ready。
- 成功后退出 0。
- 失败或超时后退出非 0。

注意：

- 这是一次性部署 job，不是常驻服务。
- 不要用 `tail -f /dev/null` 强行保持它常驻。
- Compose 应通过 `condition: service_completed_successfully` 让 gateway 等待它成功。
- 如果后续需要模型 reconcile，可另行设计 reconciler；不要把第一阶段 deployer
  临时改成隐式控制平面。

## 启动顺序

Compose 期望顺序：

1. `ray-head` 启动并健康。
2. `ray-worker` 加入 Ray cluster 并健康。
3. `serve-deployer` 部署 Serve applications。
4. `serve-deployer` 成功退出。
5. `gateway` 启动并暴露 `8000`。

启动后验证命令方向：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

真实推理 smoke 应在目标机器上执行，因为当前开发环境不能执行测试。

## 健康检查和就绪检查

### ray-head

使用：

```bash
ray status --address=127.0.0.1:6379
```

### ray-worker

使用：

```bash
ray status --address=ray-head:6379
```

这只能证明 worker 容器能连接到 cluster。更严格的 worker 资源检查后续可通过
Ray dashboard API 或 `ray status` 输出解析实现。

### serve-deployer

不建议做常驻 healthcheck。它的健康语义就是：

- exit code 0：Serve apps 已部署并通过 `run_serve_runtime.py` 的 ready 等待。
- exit code 非 0：部署失败。

### gateway

第一阶段使用：

```bash
GET /healthz
```

当前 `/readyz` 与 `/healthz` 基本一致，不应过度解释为完整下游 ready。

后续增强方向：

- gateway 能连接 Ray。
- configured local models 对应的 Serve app 存在。
- deployment 至少有 healthy replica。
- 可选检查模型 artifact 是否存在。

这些增强应另开小任务，不要和第一阶段 Compose 拆分混在一起。

## 配置和路径约定

### 容器内路径

建议统一：

```text
/app     repository code
/models  model artifacts
```

`config/settings.compose.yaml` 应使用：

```yaml
model_store:
  root_dir: /models
```

原因：

- `serve-deployer` 构建 runtime context 时会解析模型路径。
- `ray-worker` 内的 Serve replica 会使用该路径加载模型。
- 两类容器看到的路径必须一致。

### Ray address

Compose 网络内使用：

```text
ray-head:6379
```

不要在容器内依赖：

```text
auto
127.0.0.1:6379
localhost:6379
```

除非该命令运行在 `ray-head` 容器内部。

### Ray runtime_env

`scripts/run_serve_runtime.py` 当前设置了：

```python
runtime_env = {
    "py_executable": sys.executable,
    "working_dir": ".",
    "excludes": ["pyproject.toml", "uv.lock", ".venv", ".git"],
    "env_vars": {"RAY_RUNTIME_ENV_MODIFY_PYTHON_PATH": "0"},
}
```

Compose 模式下需要确认：

- `serve-deployer` 和 `ray-worker` 使用相同 image。
- `/app` 中代码一致。
- 不要把大模型权重放在 `working_dir` 下面。
- 如果 Ray runtime_env 打包导致镜像内代码和运行时分发行为冲突，应优先调整为
  镜像内代码路径一致，而不是依赖 Ray 动态构建环境。

## 未来扩展路径

第一阶段完成后，未来可以逐步演进。

### 多 gateway 容器

目标形态：

```text
nginx / lb
-> gateway-1 container
-> gateway-2 container
-> gateway-N container
```

届时建议：

- 每个 gateway 容器可设 `--workers 1` 或较小 worker 数。
- 前面加入 Nginx / Traefik / 云负载均衡。
- Prometheus 按 gateway instance 抓取 metrics。
- 重新评估 process-local admission 的全局放大效应。

### 全局 admission

当前 admission 是 process-local。多 gateway 容器后，全局实际容量大约会按实例数放大。

如果需要严格全局限制，后续可设计：

- Redis-backed shared counter。
- 独立 control-plane admission service。
- Ray-aware capacity aggregator。

这不是第一阶段内容。

### runtime worker isolation

当前 `runtime_worker_enabled: false`，相关接口仍是未来边界。

如果后续要把 Ray Serve handle 调用隔离到 runtime worker 进程或服务中，应基于
`src/infer_nexus/runtime/worker_client.py` 另行设计，不要和第一阶段 Compose 拆分混做。

## 实施顺序建议

下次实施时按以下顺序做：

1. 新建分支：

```bash
git switch -c feat/docker-compose-runtime-split
```

2. 检查工作树：

```bash
git status --short
```

如果已有用户修改的 Dockerfile / compose 文件，先确认保留策略，不要覆盖。

3. 新增 `runtime.ray_address` 配置字段。

4. 在 gateway lifespan 中，当 `execution_mode: serve` 且 `runtime.ray_address` 非空时
   初始化 Ray 连接。

5. 新增 `config/settings.compose.yaml`。

6. 新增或重建 `Dockerfile`。

7. 新增或重建 `docker-compose.yml`。

8. 更新部署文档或 `README` 中的 Compose 启动说明。

9. 只做静态检查和人工 review。当前环境不能执行测试时，不要假装测试已运行。

10. 提交：

```bash
git add <changed-files>
git commit -m "feat: add docker compose runtime split"
```

## 验证计划

当前环境无法执行测试，因此实施时最终报告应明确：

```text
Tests not run: current environment cannot execute tests.
```

在具备目标运行环境的机器上，应执行：

```bash
docker compose build
docker compose up
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

如果有可用模型和硬件，再执行项目已有 smoke：

```bash
python scripts/smoke_chat_remote.py --base-url http://127.0.0.1:8000
```

同时检查：

- `ray-head` dashboard 是否可访问。
- `ray-worker` 是否在 Ray cluster 中出现。
- Serve applications 是否处于 healthy / running。
- gateway 日志是否能成功获取 Serve deployment handle。
- Ray worker 日志中 vLLM 是否在 replica 内启动。

## 风险清单

### gateway 无法连接 Ray

症状：

- gateway 启动成功，但第一次推理请求获取 Serve handle 失败。

优先检查：

- `runtime.ray_address` 是否为 `ray-head:6379`。
- gateway 是否在 `serve` 模式启动。
- gateway lifespan 是否执行了 Ray init。
- `serve-deployer` 是否已经成功退出。

### Serve deployment 已部署但 replica 无法加载模型

症状：

- `serve-deployer` 超时或失败。
- Ray worker 日志出现模型路径不存在。

优先检查：

- `model_store.root_dir` 是否为容器内路径。
- `serve-deployer` 和 `ray-worker` 是否都挂载了同一个模型目录。
- `config/models.yaml` 中模型路径是否适配容器路径。

### Ray worker 没有可用 GPU/NPU 资源

症状：

- Serve app pending。
- Ray 报资源不足。

优先检查：

- Compose 设备映射。
- `ray start --num-gpus` 或 `--resources {"NPU": ...}`。
- `config/models.yaml` 中 `gpu_per_replica * min_replicas` 是否超过可用池。

### healthcheck 误判

症状：

- 容器 unhealthy，但进程仍在运行。
- 或 gateway healthy，但模型推理失败。

原因：

- `/healthz` 只表示 gateway 进程可响应。
- 当前 `/readyz` 尚未完整检查 Ray Serve 下游状态。

处理：

- 第一阶段接受这个限制。
- 后续单独增强 `/readyz`。

## 完成标准

第一阶段完成后应满足：

- 有一个明确的 runtime image 构建入口。
- Compose 能表达 `gateway`、`ray-head`、`ray-worker`、`serve-deployer` 四类容器。
- `serve-deployer` 是一次性 job，成功后退出。
- `gateway` 容器内支持多个 Uvicorn worker。
- gateway 通过配置连接 Compose 网络中的 `ray-head`。
- 模型路径在 `serve-deployer` 和 `ray-worker` 中一致。
- 文档明确第一阶段不引入 Nginx、不拆多个 gateway 容器、不把 Serve deployment
  做成 Compose service。
- 代码提交在独立分支。
- 最终说明明确测试未运行及原因。
