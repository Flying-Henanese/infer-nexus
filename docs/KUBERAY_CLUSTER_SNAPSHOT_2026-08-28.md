# KubeRay 集群状态快照（更新于 2026-08-31）

> 文件名保留为 `KUBERAY_CLUSTER_SNAPSHOT_2026-08-28.md` 以追溯历史；本文内容由 2026-08-31 的现场状态替代。  
> 采集时间：2026-08-31 19:06–19:08（Asia/Shanghai，UTC+08:00）  
> 采集入口：T4 control panel（`ssh t4`，`192.168.0.67`）  
> 性质：只读的线上状态快照，不是可直接 `apply` 的部署清单。

## 一句话结论

三节点 Kubernetes、NVIDIA device plugin 和 KubeRay 均健康。Kubernetes 共上报 **16 张可分配 NVIDIA GPU**：两台 A100 各 4 张，T4 control-plane 节点 8 张。

`ray/infer-nexus-ray` 已有 **5 个 A100 Ray worker**（A100-194 上 4 个、A100-213 上 1 个），向 Ray 注册 **42 CPU、168 GiB 内存、5 GPU**。A100-213 已恢复为 `DiskPressure=False`；五个预注册模型的 Ray Serve application 都是 `RUNNING / HEALTHY`，没有待调度请求。

北向 Gateway 已运行在 T4 节点，通过 NodePort **`192.168.0.67:30800`** 暴露 API。已通过该端口逐一验证聊天、向量与重排的五个模型，全部返回 HTTP 200 且响应结构正确。

## 当前拓扑

```text
客户端
  |
  | http://192.168.0.67:30800
  v
T4 bms-v38f-0004（control-plane + worker，T4 x8）
  - infer-nexus-gateway（CPU-only，1/1 Ready）
  - NodePort 8000 -> 30800
  - t4-inference-workers：存在但 replicas=0
  |
  | ray://infer-nexus-ray-head-svc.ray.svc.cluster.local:10001
  v
A100 bms-3qab-0001-0001（A100 80GB x4）
  - Ray head + 4 x a100-workers
  - KubeRay Operator
  |
  +-----------------------> A100 bms-3qab-0001-0003（A100 80GB x4）
                             - 1 x a100-213-workers

Ray Serve Controller（head） -> 5 个独立 model application，均 HEALTHY
```

Flannel Pod CIDR：T4 `10.244.0.0/24`、A100-194 `10.244.1.0/24`、A100-213 `10.244.2.0/24`；网络后端为 VXLAN。

## Kubernetes 节点与 GPU

三个节点均为 `Ready`、未 cordon，未发现内存、磁盘或 PID 压力。

| 节点 | 角色 / 内网 IP | 系统与 containerd | 污点 | GPU capacity / allocatable | 当前 infer-nexus 工作负载 |
|---|---|---|---|---:|---|
| `bms-v38f-0004` | control-plane, worker / `192.168.0.67` | Ubuntu 22.04.5；containerd 1.7.27 | `node-role.kubernetes.io/control-plane:NoSchedule` | T4 `8 / 8` | Gateway；无 Ray worker |
| `bms-3qab-0001-0001` | worker / `192.168.0.194` | Ubuntu 20.04.6；containerd 1.7.18 | 无 | A100 `4 / 4` | Ray head + 4 个 worker |
| `bms-3qab-0001-0003` | worker / `192.168.0.213` | Ubuntu 22.04.5；containerd 1.7.27 | 无 | A100 `4 / 4` | 1 个 worker |

Kubernetes/kubelet 版本均为 `v1.34.10`。NVIDIA device plugin DaemonSet（`nvidia-device-plugin-daemonset`）为 3/3 Ready；两台 A100 的 GPU allocatable 都为 4。

## KubeRay 与 RayCluster

- KubeRay Operator：`default/kuberay-operator` Deployment，`1/1 Available`，镜像 `m.daocloud.io/quay.io/kuberay/operator:v1.6.2`。
- 已安装 `RayCluster`、`RayJob`、`RayService` CRD；当前只有一个 `RayCluster`，没有 `RayJob` 或 `RayService` 对象。
- Ray 版本：`2.55.1`。

### `ray/infer-nexus-ray`

| 项目 | 当前值 |
|---|---|
| 创建时间 / generation | `2026-08-31T10:33:47Z` / `2` |
| 状态 | `ready` |
| worker 期望 / Available / Ready | `5 / 5 / 5` |
| Ray 总资源 | 42 CPU、168 GiB memory、5 GPU |
| 采集时资源用量 | 26 / 42 CPU、2.5 / 5 GPU；0 / 168 GiB Ray memory |
| head Pod | `infer-nexus-ray-head-l2qpm`，`10.244.1.37`，位于 A100-194 |
| head Service | `infer-nexus-ray-head-svc`，headless（`ClusterIP=None`） |

head Service 的集群内端口：`10001`（Ray Client）、`8265`（Dashboard）、`6379`（GCS）、`8080`（metrics）、`8000`（Serve）。Gateway 使用稳定 DNS 名连接，不依赖 Pod IP。

| worker group | replicas / min / max | 节点 | 单 Pod GPU limit | 当前状态 |
|---|---:|---|---:|---|
| `a100-workers` | 4 / 1 / 4 | `bms-3qab-0001-0001` | 1 | 4 Running |
| `a100-213-workers` | 1 / 0 / 4 | `bms-3qab-0001-0003` | 1 | 1 Running |
| `t4-inference-workers` | 0 / 0 / 8 | `bms-v38f-0004` | 1 | 未启用 |

本次恢复先在 A100-194 启动 4 个 worker；A100-213 清理磁盘并恢复健康后再启用 1 个 worker。当前五个模型副本会占用不同的 Ray worker，因此保留 5 个 1-GPU worker；不要在未重新规划 placement 前缩回 A100-194 的第 4 个 worker。

所有 Ray head/worker 使用 `infer-nexus:runtime` 且 `imagePullPolicy: Never`。T4 直接确认有该镜像；两台 A100 上运行中的 Ray Pod 也以 `Never` 使用此镜像，因此它们的本地镜像缓存是启动前提。

## Ray Serve 与模型状态

`serve status` 显示 `proxies: {}`；业务请求由 Gateway 经 Ray Serve DeploymentHandle 分发。五个 application 都有一个 `RUNNING` replica，deployment 全部 `HEALTHY`。

| Gateway alias | Ray Serve application / deployment | 任务 | 状态 | 单副本 Ray 资源 |
|---|---|---|---|---:|
| `qwen3.5-27b` | `infer-nexus-model-Qwen3.5-27B` / `model-Qwen3.5-27B` | chat | RUNNING / HEALTHY | 8 CPU、1.0 GPU |
| `mineru` | `infer-nexus-model-MinerU2.5-Pro-2604-1.2B` / `model-MinerU2.5-Pro-2604-1.2B` | chat / vision | RUNNING / HEALTHY | 4 CPU、0.4 GPU |
| `qwen3-embedding-8b` | `infer-nexus-model-qwen3-embedding-8b` / `model-qwen3-embedding-8b` | embedding | RUNNING / HEALTHY | 2 CPU、0.3 GPU |
| `bge-reranker` | `infer-nexus-model-bge-reranker-large` / `model-bge-reranker-large` | rerank | RUNNING / HEALTHY | 4 CPU、0.2 GPU |
| `qwen3.5-9b` | `infer-nexus-model-Qwen3.5-9B` / `model-Qwen3.5-9B` | chat / vision | RUNNING / HEALTHY | 8 CPU、0.6 GPU |

最小副本合计预约 26 CPU、2.5 GPU，和 `ray status` 的实际用量一致。分数 GPU 是 Ray 的逻辑调度额度，不是 Kubernetes 设备分片或显存隔离。

## Gateway 与北向暴露

| 项目 | 当前值 |
|---|---|
| Deployment | `ray/infer-nexus-gateway`，1 / 1 Ready |
| Pod | `infer-nexus-gateway-7df48bcb94-dqgk4`，`10.244.0.8` |
| 调度节点 | T4 `bms-v38f-0004` |
| 节点选择 / 容忍 | 固定 hostname；容忍 control-plane `NoSchedule` 污点 |
| 镜像 | `infer-nexus:runtime`，`imagePullPolicy: Never` |
| Service | `ray/infer-nexus-gateway`，NodePort |
| Service 地址 | ClusterIP `10.110.137.210:8000`；NodePort `192.168.0.67:30800` |
| 配置 | `config/settings.k8s.yaml`；`INFER_NEXUS_RAY_ADDRESS=ray://infer-nexus-ray-head-svc.ray.svc.cluster.local:10001` |

Gateway 是 CPU-only 的独立 Deployment，不是 Ray node，也不直接暴露 Ray Dashboard、GCS 或 Serve HTTP proxy。目前没有 Ingress；NodePort 的集群外可达性仍取决于主机防火墙和网络安全策略，本文只验证了 T4 本机经 NodePort 的链路。

### Gateway 冒烟测试（NodePort 30800）

下列请求均通过 `http://127.0.0.1:30800` 发出，经过 NodePort、Gateway、Ray Client、Ray Serve 与模型 replica：

| 模型 | 请求 | 校验 | 结果 |
|---|---|---|---|
| `qwen3.5-27b` | `POST /v1/chat/completions` | 模型名及一个 choice | PASS，HTTP 200 |
| `mineru` | `POST /v1/chat/completions` | 合法 chat completion 响应 | PASS，HTTP 200 |
| `qwen3-embedding-8b` | `POST /v1/embeddings` | 一条非空 embedding 向量 | PASS，HTTP 200 |
| `bge-reranker` | `POST /v1/rerank` | 一条含数值 score 的 top-1 结果 | PASS，HTTP 200 |
| `qwen3.5-9b` | `POST /v1/chat/completions` | 模型名及一个 choice | PASS，HTTP 200 |

`GET /healthz`、`GET /readyz` 与 `GET /v1/models` 也都返回 200；模型目录正好为上表五个 alias。

## NAS 共享代码与配置

Pod 的只读 hostPath 挂载为：

- `/nas_data/zhousj/infer-nexus` -> `/workspace/infer-nexus`
- `/nas_data/models` -> `/models`

这是固定节点实验环境的 POC 形态：每个将承载 Gateway 或 Ray worker 的节点都必须具备同一路径和匹配内容；`hostPath` 不具备 PVC 的跨节点存储与调度语义。

采集时 NAS 共享仓库为 commit `41dbdb0`，并存在未提交配置/部署/快照文件：`config/models.yaml`、`config/settings.yaml`、`config/settings.k8s.yaml`、`deploy/`、本文档。关键文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `config/models.yaml` | `788ea2ea200db7110aedd066ab45a13784e9f5764e1a001ca3f1d2debf116705` |
| `config/settings.k8s.yaml` | `39413b953823e748d851b487d2021705a828d1764506db3e59b16226635f5de9` |
| `deploy/kuberay/gateway.yaml` | `4cff646669bfb1df741fcc3aa3dd5ae23e0fb9967fd5d12172ee4da40906b38b` |
| `deploy/kuberay/raycluster.recovery.yaml` | `5a0964f6cc39314712d7f21ae74ee232f8977a7afab9a270243a6370d8dde5f9` |
| `deploy/kuberay/serve-deployer.job.yaml` | `be6381bbeeb0fabb9a479a520f7b3fa26109d82446d12af2c9218b8ab6ea637b` |

`raycluster.recovery.yaml` 仅管理 RayCluster 拓扑与 worker 副本；`serve-deployer.job.yaml` 是一次性提交 Serve 应用的 RayJob，不能与日常扩缩容清单一起 apply，否则会触发不必要的模型 replica 更新。采集时没有运行中的 RayJob 或 RayService。

## 当前边界与风险

1. **T4 尚未加入 Ray 推理池。** `t4-inference-workers.replicas=0`，当前只承载 Gateway。若启用该 worker group，模型尚无模型级硬件型号约束，不能假设大模型一定留在 A100。
2. **镜像与存储是节点前置条件。** `imagePullPolicy: Never`、节点本地镜像与 `hostPath` 使新节点、重建与故障恢复依赖人工预置；后续应使用可审计镜像仓库与共享存储/PVC。
3. **没有常驻 RayJob、RayService 或 Ingress。** 已保留一次性 `serve-deployer.job.yaml` 用于重新提交目录中的模型；它完成后应删除。当前动态 Serve 应用仍不受 RayService 生命周期管理；单节点 NodePort 适合内网 POC，不是生产入口。
4. **Ray head 为单点。** head 在 A100-194；该节点或 head 重启会影响整个 Ray 集群和 Gateway 后端请求。
5. **配置发布需受控。** ConfigMap/镜像化配置与 revision 尚未引入。直接修改 NAS hostPath 中的文件不会让已运行的 Gateway 或 Serve actor 自动、可审计地重载。
6. **Gateway 扩容会改变准入容量。** 当前只有 1 个 Uvicorn worker，现有 admission 限制是进程本地；扩为多个 Pod/worker 前应先定义全局限流语义。

## 复查命令（只读）

```bash
ssh t4 'kubectl get nodes -o wide'
ssh t4 'kubectl get rayclusters,rayservices,rayjobs -A -o wide'
ssh t4 'kubectl -n ray get pods,svc -o wide'
ssh t4 'kubectl -n ray get deployment infer-nexus-gateway -o wide'
```

```bash
ssh t4 'H=$(kubectl -n ray get pods -l ray.io/node-type=head -o jsonpath="{.items[0].metadata.name}"); kubectl -n ray exec "$H" -c ray-head -- serve status'
ssh t4 'H=$(kubectl -n ray get pods -l ray.io/node-type=head -o jsonpath="{.items[0].metadata.name}"); kubectl -n ray exec "$H" -c ray-head -- ray status'
ssh t4 'curl -fsS http://127.0.0.1:30800/v1/models | jq'
```

## 证据范围

本文基于 2026-08-31 19:06–19:08 从 T4 control panel 执行的只读 `kubectl get`、`kubectl exec ... serve status`、`ray status`、Gateway HTTP 冒烟请求和 NAS 共享代码检查。本文档更新本身没有修改 RayCluster、Ray Serve application、节点标签/污点、模型权重或设备配置；Gateway 与 Service 已在此前的部署步骤中创建。

---

## 2026-09-01 变更附录：开启 KubeRay 弹性伸缩

> 背景：当日早间对 `qwen3.5-9b` 的并发压测暴露了两个问题——Serve 扩容决策触发后，新副本因没有任何节点能提供整 8 空闲 CPU 而无法落地（挂起 10 分钟后被缩容回收）；且集群未开启自动扩缩容，worker 数只能手动调整。据此实施以下变更。

### 变更内容

- `deploy/kuberay/raycluster.recovery.yaml` 增加 `enableInTreeAutoscaling: true`；`a100-213-workers` 调整为 `replicas=2, minReplicas=2, maxReplicas=4`（温备基线，保证 9B 扩容时资源现成）；`a100-workers` 维持 `min=1/max=4`；`t4-inference-workers` 维持禁用（T4 16GB 无法承载已注册大模型，且模型配置尚无硬件型号约束）。
- 切换 `enableInTreeAutoscaling` 需重建集群：删除并重建 RayCluster（新 head `infer-nexus-ray-head-qxhr6`，含 autoscaler 容器），随后用一次性 `serve-deployer.job.yaml` 重新提交 5 个 Serve 应用（约 17 分钟全部就绪），完成后已删除该 RayJob。
- **Gateway 需手动重启**：集群重建后 Gateway 进程持有指向旧 head 的 Ray Client 会话，推理请求全部挂起（日志出现 `Ray Client is not connected` 与旧 ObjectRef 报错），而 `/healthz`、`/readyz` 不经过推理路径仍返回 ok。执行 `kubectl -n ray rollout restart deployment infer-nexus-gateway` 后恢复，现运行 `infer-nexus-gateway-6b57bb4ccd-kbrrd`。
- 集群重建后 autoscaler 会立即把空闲 worker 收敛到各组的 minReplicas（重建初期 a100-workers 曾被从 4 收敛到 1，随后随应用部署的资源需求自动回升）。

### 弹性伸缩实测（qwen3.5-9b，8 路并发冒烟负载）

| 阶段 | 时间（本地） | 事件 |
|---|---|---|
| 扩容决策 | 18:47:18 | `Upscaling from 1 to 2 replicas. Current ongoing requests: 4.00` |
| 节点扩容 | 18:47:21 | autoscaler `Adding 1 node(s) of type a100-workers`（新 worker `7mvhb`，A100-194） |
| 副本就绪 | 18:50:36 | 副本 `bg6rosea` 在新 worker 上启动成功（vLLM 引擎初始化 175s），双副本 RUNNING/HEALTHY |
| 缩容决策 | 19:01:08 | 负载 18:52:28 结束 + 600s `downscale_delay_s` 到期，`Downscaling ... ongoing requests: 0.00` |
| 节点回收 | 19:02:13 | 旧副本 `2jwf26kn` 停止后，其所在空闲 worker `sms6w` 被 autoscaler 回收 |

- 全链路扩容时延约 4 分钟（决策 ~60s + 节点拉起 ~30s + 引擎初始化 ~200s）；缩容全程约 10 分钟（由 downscale_delay 主导）。
- 缩容后 9B 常驻副本换为 `bg6rosea`（`7mvhb`，A100-194）；稳态 4 worker（A100-194 ×2、A100-213 ×2）、26/34 CPU、2.5/4 GPU，与五个模型最小副本 footprint 一致。
- 结构性约束未变：每 worker 8 CPU 恰好容纳一个 8-CPU 副本（27B/9B），大模型副本数与 worker 数 1:1 锁定；两组 worker 合计上限 8 个 1-GPU worker。worker 形态（CPU:GPU 配比）是后续更根本的优化方向。

### 遗留问题（本次未处理）

1. 流式接口 100% 失败：Gateway 经 Ray Client 连接时 `DeploymentHandle.options(stream=True)` 抛 `RuntimeError`（Ray 2.55 的 Ray Client 限制），`executor.py` 静默回退为非流式调用，replica 端抛 TypeError 后 Gateway 返回 HTTP 500。修复方向：流式改走 Serve HTTP ingress 透传，或副本方法改为返回完整分块由 Gateway 伪流式转发。
2. Gateway 的 `/readyz` 不校验 Ray 连接健康度（集群重建后仍返回 ok），建议增强，避免假就绪。
3. 模型无硬件型号约束；在解决约束之前不得启用 T4 worker 组。
