# Ascend 平台部署配置说明

本文用于指导在 Ascend NPU 平台上部署 infer-nexus。当前部署方式使用 Docker Compose 管理 ray-head、ray-worker 和 serve-deployer；公网 API 由 ray-head 上的 Ray Serve Gateway ingress 提供。

## 一、需要配置的文件

| 文件 | 作用 |
| --- | --- |
| config/settings.ascend-compose.yaml | Ascend 平台、Ray、运行时和模型存储配置 |
| config/models.yaml | 注册模型、模型路径、任务类型和副本资源配置 |
| Shell 环境变量 | 宿主机路径、NPU 可见设备和镜像参数 |
| ascend_deploy/docker-compose.yml | 容器编排、NPU 设备和 CANN 路径挂载 |
| ascend_deploy/dockerfile | Ascend 基础镜像和应用依赖安装方式 |

## 二、宿主机和基础镜像

部署前确认：

- 宿主机已经安装 Ascend 驱动、CANN 和对应固件。
- npu-smi info 可以看到目标 NPU。
- Docker 和 Docker Compose v2 可用。
- Ascend 基础镜像已经存在，默认镜像名为 mineru:npu-latest。
- 基础镜像中的 Python、PyTorch、torch-npu、vLLM 和 vllm-ascend 版本相互兼容。
- 宿主机模型目录中已经准备好要部署的模型。

建议检查：

~~~bash
npu-smi info
docker version
docker compose version
docker image inspect mineru:npu-latest
~~~

## 三、config/settings.ascend-compose.yaml

启动时必须使用 config/settings.ascend-compose.yaml。

### 3.1 服务和目录配置

~~~yaml
service:
  name: infer-nexus
  host: 0.0.0.0
  port: 8000
  workers: 2

catalog:
  models_path: config/models.yaml

model_store:
  root_dir: /models
  huggingface_endpoint: https://hf-mirror.com
~~~

说明：

- host 应为 0.0.0.0，宿主机才能访问容器服务。
- port 默认是 8000，对应 Compose 中的 8000:8000。
- root_dir 是容器内路径，不是宿主机路径。
- Compose 会把宿主机的 MODEL_STORE_HOST_PATH 挂载到容器的 /models。

### 3.2 Ascend 和 Ray 配置

~~~yaml
cluster:
  inference_device_type: npu
  device_pool_boundary: ray_runtime_visible_devices
  default_platform: npu
~~~

必须确认 inference_device_type 和 default_platform 都是 npu，不要误用 CUDA 的 settings 文件。

### 3.3 真实推理运行时

~~~yaml
runtime:
  backend: vllm
  execution_mode: serve
  backend_init_mode: real
  device_env_strategy: ray_managed
~~~

真实部署不能使用 execution_mode: stub 或 backend_init_mode: stub，否则服务可能启动但不会加载真实模型。

### 3.4 超时和并发

初次部署可以先使用默认值：

~~~yaml
runtime:
  serve_request_timeout_seconds: 120
  serve_stream_idle_timeout_seconds: 30
  serve_stream_max_lifetime_seconds: 900
  max_inflight_per_model: 4
  max_queued_per_model: 0
  circuit_breaker_enabled: false
~~~

第一次实验不建议同时提高并发、模型副本数和上下文长度，应先完成单模型推理。

## 四、部署环境变量

不需要创建额外的环境变量文件。直接在启动 Docker Compose 的 Linux Shell 中导出变量即可：

~~~bash
export ASCEND_BASE_IMAGE=mineru:npu-latest
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_STORE_HOST_PATH=/data/models
export APP_UID=10001
export APP_GID=10001
export GATEWAY_WORKERS=2
export SERVE_READY_TIMEOUT_SECONDS=900
~~~

其中 ASCEND_RT_VISIBLE_DEVICES 是必须提供的变量；其他变量可以使用 Compose 或 Dockerfile 中的默认值，但生产部署建议显式设置模型目录和基础镜像。

### 4.1 必填配置

#### ASCEND_BASE_IMAGE

Ascend 基础镜像名称。如果更换镜像，必须确认其中已经包含兼容的 Python、PyTorch、torch-npu、vLLM 和 vllm-ascend。不要让项目依赖安装覆盖镜像内置的 NPU 运行时。

#### ASCEND_RT_VISIBLE_DEVICES

指定 infer-nexus 使用的 NPU 池：

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
~~~

该变量同时影响容器和 Ray 可见的 NPU、Ray 注册的 NPU 资源数量，以及模型副本是否能够被调度。Compose 缺少该变量时 ray-worker 会退出。

#### MODEL_STORE_HOST_PATH

宿主机模型目录：

~~~bash
export MODEL_STORE_HOST_PATH=/data/models
~~~

映射关系：

~~~text
/data/models -> /models
~~~

#### APP_UID 和 APP_GID

容器默认以非 root 用户 infer-nexus 运行。模型目录至少要可读，日志目录需要可写。遇到权限错误时，调整这两个值或修改宿主机目录权限。

### 4.2 可选配置

~~~bash
export GATEWAY_WORKERS=2
export SERVE_READY_TIMEOUT_SECONDS=900
export INFER_NEXUS_RAY_ADDRESS=ray-head:6379
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
~~~

通常不需要修改 INFER_NEXUS_RAY_ADDRESS，Compose 默认会通过 ray-head 服务名连接 Ray。
## 五、config/models.yaml

每个模型需要配置模型身份、任务、后端、模型路径和资源需求。

### 5.1 最小本地模型配置

~~~yaml
models:
  - name: mineru-demo
    alias: mineru-demo
    task: chat
    backend: vllm
    compat_mode: vllm_native
    model_path: opendatalab/MinerU2.5-Pro-2604-1.2B
    dtype: auto
    tensor_parallel_size: 1
    max_model_len: 8192
    cpu_per_replica: 4
    gpu_per_replica: 1
    gpu_memory_utilization: 0.8
    min_replicas: 1
    max_replicas: 1
    vllm:
      engine_kwargs:
        trust_remote_code: true
      openai_serving:
        enabled: true
~~~

### 5.2 模型路径和容器挂载

推荐使用相对于 /models 的路径：

~~~yaml
model_path: Qwen/Qwen3.5-9B
~~~

对应关系：

~~~text
宿主机：/data/models/Qwen/Qwen3.5-9B
容器：  /models/Qwen/Qwen3.5-9B
~~~

不要把宿主机路径直接写进 model_path。当前 Compose 挂载的是 /models，不是 /app/models。因此下面这种写法通常是错误的：

~~~yaml
model_path: /app/models/Qwen/Qwen3.5-9B
~~~

### 5.3 重要模型字段

| 配置项 | 说明 |
| --- | --- |
| name | 模型规范名称 |
| alias | 客户端请求时使用的名称 |
| task | chat、embedding 或 rerank |
| backend | 本地模型使用 vllm |
| compat_mode | Chat/Embedding 通常使用 vllm_native |
| model_path | 容器内模型路径，建议相对 /models 配置 |
| dtype | 例如 auto、float16、bfloat16 |
| tensor_parallel_size | 模型张量并行使用的设备数量 |
| max_model_len | 最大上下文长度 |
| cpu_per_replica | 每个副本申请的 CPU 数量 |
| gpu_per_replica | Ascend 分支中映射为每个副本的 NPU 资源数量 |
| gpu_memory_utilization | vLLM 显存利用率 |
| min_replicas | 最小副本数，启动时会加载这些副本 |
| max_replicas | 最大副本数 |
| vllm.engine_kwargs | vLLM 引擎参数 |
| vllm.openai_serving.enabled | vllm_native Chat/Embedding 必须开启 |

### 5.4 gpu_per_replica 不要改名

虽然运行平台是 NPU，但当前配置字段仍然是：

~~~yaml
gpu_per_replica: 1
~~~

Ascend settings 下，代码会把它转换成：

~~~python
{"resources": {"NPU": 1}}
~~~

不要自行写成不支持的字段：

~~~yaml
npu_per_replica: 1
~~~

### 5.5 第一次实验只启用一个模型

当前 config/models.yaml 中有多个启用模型，并且大多数模型的 min_replicas 为 1。Serve 启动时会尝试同时部署多个模型。

如果学生只有一两张 NPU，建议：

- 先注释掉其他模型。
- 只保留一个较小模型。
- min_replicas: 1。
- max_replicas: 1。
- tensor_parallel_size: 1。
- gpu_per_replica: 1。

完成单模型推理后，再逐步增加模型和副本。

## 六、Ascend 设备和 CANN 路径

当前 Compose 使用以下设备节点：

~~~text
/dev/davinci_manager
/dev/devmm_svm
/dev/hisi_hdc
/dev/davinci*
~~~

并挂载以下宿主机路径：

~~~text
/usr/local/Ascend/driver
/usr/local/dcmi
/usr/local/bin/npu-smi
/usr/local/sbin
/usr/local/Ascend/driver/tools/hccn_tool
~~~

如果学生机器的 CANN 安装路径不同，需要同步调整 ascend_deploy/docker-compose.yml。ray-worker 当前使用 privileged: true，Docker 环境必须允许特权容器。

## 七、Dockerfile 和依赖

ascend_deploy/dockerfile 会使用 Ascend 基础镜像，并安装应用依赖，但会排除镜像已经提供的 NPU 运行时包。

不要随意覆盖以下包的版本：

~~~text
torch
torch-npu
torchaudio
torchvision
transformers
vllm
vllm-ascend
~~~

如果构建失败，优先检查基础镜像、APT/PyPI 网络和依赖版本兼容性。

## 八、启动命令

从仓库根目录执行。先检查 Compose 展开结果：

~~~bash
docker compose -f ascend_deploy/docker-compose.yml config
~~~

确认无误后构建并启动：

~~~bash
docker compose -f ascend_deploy/docker-compose.yml build
docker compose -f ascend_deploy/docker-compose.yml up
~~~

当前 Compose 内部已经指定 config/settings.ascend-compose.yaml，不需要额外传递 settings 参数。

如果不使用 Compose，而是直接运行启动脚本：

~~~bash
bash scripts/start_minimal_ascend.sh --settings config/settings.ascend-compose.yaml --ascend-visible-devices 0,1,2,3
~~~

## 九、启动后的验证

### 9.1 查看容器状态

~~~bash
docker compose -f ascend_deploy/docker-compose.yml ps
~~~

预期状态：

- ray-head：运行中且健康。
- ray-worker：运行中且健康。
- serve-deployer：成功执行后退出，退出码应为 0。
- `ray-head:8000` 的 `/healthz`、`/readyz` 和 `/v1/models` 可访问。

### 9.2 查看日志

~~~bash
docker compose -f ascend_deploy/docker-compose.yml logs -f ray-head ray-worker serve-deployer
~~~

重点检查 Ray 的 NPU 资源、模型目录、vLLM 初始化和 Serve 健康状态。

### 9.3 API 验证

~~~bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models

curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"你好，请用一句话介绍 Ray Serve。"}],"max_tokens":64}'
~~~

模型名称应使用 /v1/models 返回的实际 alias 或 served model name。

也可以运行冒烟脚本：

~~~bash
python scripts/smoke_chat_remote.py --base-url http://127.0.0.1:8000/v1 --model qwen3.5-9b --num-requests 1 --concurrency 1
~~~

## 十、常见问题

| 现象 | 优先检查 |
| --- | --- |
| 缺少 ASCEND_RT_VISIBLE_DEVICES | 是否已在当前 Shell 中 export ASCEND_RT_VISIBLE_DEVICES |
| ray-worker 退出 | NPU 设备节点、CANN 路径和 npu-smi info |
| model_artifact_missing | MODEL_STORE_HOST_PATH、/models 和 model_path |
| Serve 资源不足 | NPU 可见数量、gpu_per_replica 和副本数 |
| vLLM 初始化失败 | 基础镜像、CANN、torch-npu、vLLM 和 vllm-ascend 版本 |
| Gateway 健康但推理失败 | serve-deployer 日志和模型配置 |
| 容器权限错误 | APP_UID、APP_GID 及目录权限 |

## 十一、首次部署验收标准

- 宿主机和容器内可以看到目标 NPU。
- Ray Worker 成功注册预期的 NPU 资源。
- serve-deployer 以退出码 0 完成。
- Serve Gateway ingress 通过健康检查。
- /v1/models 返回预期模型。
- 至少一个 Chat 请求成功。
- 日志中没有持续的模型加载、资源不足或 NPU 初始化错误。
