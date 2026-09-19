# 类似系统的日志架构及 infer-nexus 可借鉴实践

日期：2026-09-19。性质：调研结论，不代表已经实施。

本报告比较 Docker/Kubernetes、Ray/KubeRay、OpenTelemetry、Grafana Loki、NVIDIA Triton 和 KServe 的官方做法。KubeRay 只作为未来多节点环境参考；当前单机版本不需要引入 Kubernetes 或 KubeRay。

## 共同架构

现实系统很少让一个组件完成全部日志职责，通常分成五层：

```text
应用与框架生成
      ↓
stdout / 本地文件
      ↓
节点采集器
      ↓
独立存储与留存
      ↓
查询、关联、告警
```

生成者决定事件语义和敏感信息边界；容器运行时或文件 writer 负责本地可靠落地；采集器发现、解析、补充节点身份并保存读取进度；后端负责跨节点留存和索引；查询层提供面向人的降噪视图。日志存储的生命周期在生产集群中通常独立于容器和节点。

这一分层比选择 Loki 或 Elasticsearch 更重要。单机版也可保留相同边界，只把采集、存储和查询压缩成本地文件与 CLI。

## 典型系统如何做

### Docker 和 Kubernetes：普通容器优先 stdout/stderr

Docker 只捕获进程的 stdout/stderr；应用自行写入 bind mount 的文件不属于 `docker logs`。Docker 推荐一般场景使用带自动轮转的 `local` driver，Kubernetes 则由容器运行时和 kubelet管理容器输出及轮转。Kubernetes 官方把 stdout/stderr 称为最容易且最常采用的容器日志方式。

集群里通常每个节点运行一个采集 agent，而不是让每个业务进程直接同步发送远端日志。agent 将容器输出、系统记录和必要的文件记录发送到独立后端。只有特殊格式或只能写文件的组件才更常用 sidecar；Kubernetes 明确指出“文件再由 sidecar 输出到 stdout”会造成双份本地存储，单一流可以直接写 stdout 时不值得这样做。

来源：[Kubernetes Logging Architecture](https://kubernetes.io/docs/concepts/cluster-administration/logging/)、[Docker logging drivers](https://docs.docker.com/engine/logging/configure/)、[Docker container logs](https://docs.docker.com/engine/logging/)。

**对本项目的含义：** `ray start` 和部署器的主进程输出长期更适合交给 Docker；当前 `run_with_log.sh` 与 `container.log` 是满足现有宿主机文件约定的兼容层，不应继续扩展为日志平台。迁移必须连同架构约定和启动输出归档一起处理，不能直接删除。

### Ray/KubeRay：框架日志天然是文件，因此需要第二条采集链路

Ray 的应用和系统日志分散在每个节点的 `/tmp/ray/session_*/logs`。Ray 不提供持久日志存储，也不保证日志目录布局向后兼容。KubeRay 官方给出两种采集策略：每个 Ray Pod 放一个 Fluent Bit sidecar，或者每个节点部署 DaemonSet 并把 `/tmp/ray` 映射到 hostPath；两者都由采集器读取 Ray 文件并发送 Loki 等后端。

Ray 文档也不建议用 `RAY_LOG_TO_STDERR=1` 强制变成纯 stderr 方案，因为这会改变或关闭部分 worker 日志转发能力。Ray 属于普通“容器只写 stdout”规则的合理例外。

这说明类似 Ray 的系统并不强求“只用容器 stdout”。实际是双入口：容器输出处理启动和容器生命周期，Ray 文件处理 controller、worker、Serve deployment 及底层故障。采集后才形成统一查询视图。

来源：[Persist KubeRay logs](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/persist-kuberay-custom-resource-logs.html)、[Ray logging guide](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html)、[Ray Serve LoggingConfig](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.schema.LoggingConfig.html)。

**对本项目的含义：** 保留 `/tmp/ray` 挂载是合理的。独立 `events-*.jsonl` 也合理，因为它为 infer-nexus 提供稳定 schema 和权威事件源，避免依赖 Ray 文件名或 stdout 转发副本。未来 KubeRay 采集器只需增加该文件 glob，不要求现在采用 KubeRay。

### OpenTelemetry：统一的是记录模型，不要求统一输出介质

OpenTelemetry 同时承认 stdout、文件和 OTLP 直传。对已有应用，官方建议 Collector 的 `filelog` receiver 或 Fluent Bit 读取文件、处理轮转、发现新文件并保存 offset；结构化 JSON 比自由文本更可靠。Collector 可以在采集时补充 host、container、service 等资源身份。

OTLP 直传能够省掉文件解析，但让应用依赖网络采集路径，也失去简单本地检查的优点。对于推理服务，不应让请求路径同步依赖远端日志平台。已有 `request_id` 是应用关联键；它不自动等同于 W3C trace ID，未来增加 tracing 时应保留两个字段。

来源：[OpenTelemetry Logs](https://opentelemetry.io/docs/specs/otel/logs/)、[OpenTelemetry Log Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/)。

**对本项目的含义：** 现在不需要接入 OpenTelemetry SDK。JSONL 字段应可映射到 OTel，文件 writer 与未来采集器解耦。`schema_version`、服务、节点、进程实例和事件名应稳定。

### Loki：来源身份进入索引，请求身份留在记录中

Loki 以低基数 label 划分流。官方明确不建议把 request ID、trace ID、用户 ID、IP、Pod 名和短生命周期 instance ID作为索引 label；它们会产生大量日志流。适合 label 的是环境、集群、服务和组件等取值有限且长期稳定的身份。高基数字段应放在结构化 metadata 或 JSON 内容里按需查询。

来源：[Loki labels and cardinality](https://grafana.com/docs/loki/latest/get-started/labels/cardinality/)、[Loki label guidance](https://grafana.com/docs/loki/latest/get-started/labels/)。

**对本项目的含义：** 本地 CLI 可以按所有字段过滤；未来进入集中平台时，`service/environment/source` 可作为索引维度，`request_id/process_instance/replica_id` 留在记录内容中。

### 推理服务：常态日志与临时深度诊断分开

NVIDIA Triton 默认提供 info/warning/error 与独立 verbose level，可以在运行时读取和修改日志设置；详细 request trace 还有单独的采样率、数量和输出配置。这反映了推理系统的常见做法：常态运行保留低成本运营事件，深度执行细节只在有限时间或采样范围内打开，而不是永久打印所有内部过程。

KServe 的 inference logger 可以另行捕获请求和响应，用于审计或模型分析。它属于数据记录链路，并不等同于服务运维日志。对于 LLM，prompt、messages、token 和文档存在明显敏感性及体积问题，不应因为“日志完整”混入普通诊断日志。

来源：[Triton Logging Extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_logging.html)、[Triton trace](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2270/user-guide/docs/user_guide/trace.html)、[KServe Inference Logger](https://kserve.github.io/website/docs/model-serving/predictive-inference/logger)。

**对本项目的含义：** 保持正常事件、框架原始日志、临时 debug/trace、推理数据四类边界。当前不记录请求正文是正确的；未来若有审计或数据分析需求，应另建有权限、保留期限和脱敏规则的管道。

### 故障降级：日志链路不能拖住推理链路

Docker 的阻塞交付能减少丢失，但慢日志驱动会反向阻塞应用；非阻塞模式改用有限缓冲，缓冲满后会丢记录。OpenTelemetry Collector 的 exporter 同样使用有限队列、重试和可选磁盘持久队列，队列满或磁盘故障仍需通过失败指标暴露。成熟系统并不存在“永不阻塞、永不丢失、没有成本”的设置。

来源：[Docker log delivery mode](https://docs.docker.com/engine/logging/configure/#configure-the-delivery-mode-of-log-messages-from-container-to-log-driver)、[OpenTelemetry Collector exporter helper](https://github.com/open-telemetry/opentelemetry-collector/blob/main/exporter/exporterhelper/README.md)。

**对本项目的含义：** 日志写入失败不能无限阻塞请求线程。第一版应记录写入失败次数并向 stderr 限频报告；若采用异步 writer，队列必须有界，并为丢弃数量提供指标。Ray 原始文件继续作为应用事件链路失效时的诊断兜底。

## 和拟议方案的对照

| 拟议决定 | 现实系统对应做法 | 判断 |
| --- | --- | --- |
| 应用事件使用结构化 JSONL | OTel 推荐结构化记录；Ray 本身大量使用文件 | 保留 |
| 每进程单独事件文件 | 避免多进程争抢 writer，匹配节点采集发现模型 | 保留 |
| 保留完整 Ray session | KubeRay 同样专门采集 `/tmp/ray` | 保留，但查询只读相关日志 |
| CLI 默认降噪、原始模式展开 | 存储与查询视图分离 | 保留 |
| `container.log` 由 shell 重定向 | 普通容器更常用 stdout + runtime 轮转 | 作为过渡，后续迁移 |
| 现在部署 Loki/Collector | 多节点集中查询才产生主要收益 | 暂缓 |
| 全局把 Ray/vLLM 改成 WARNING | 推理系统通常保留常态 info，另控 verbose/trace | 不采用 |
| 请求 ID 作为未来 Loki label | 高基数索引反模式 | 不采用 |
| 保存 prompt/response 便于排错 | 属于独立 inference data/audit 管道 | 不采用 |

## 适合当前单机 Compose 的目标

```text
┌────────────────────────────────────────────────────────┐
│ 生成                                                   │
│ infer-nexus 事件    Ray/Serve/vLLM       主进程输出    │
└────────┬─────────────────┬──────────────────┬──────────┘
         │                 │                  │
         ▼                 ▼                  ▼
  events-*.jsonl   ray/session_*/logs   container.log（过渡）
         └─────────────────┬──────────────────┘
                           ▼
                    本地统一查询 CLI
               默认摘要 / request / infra / raw
```

当前直接采用：

1. 应用事件稳定 JSON schema，每进程独立文件，源头禁止敏感 payload。
2. Ray 原始文件继续宿主机可见，作为底层故障依据。
3. 本地 CLI 同时发现事件文件和 Ray 原始文件，展示与保存分离。
4. 每类来源独立轮转和容量统计；历史 session、退出进程文件另设留存。
5. 正常级别不急于减少，先统计 logger/事件频率和重复来源，再做定点治理。

现在预留、无需安装组件：

1. 字段能映射到 OTel resource/log attributes。
2. 文件命名支持采集器 glob，记录保留实际节点和来源路径。
3. stdout 与文件不承担互相矛盾的权威定义。
4. 查询条件区分低基数资源身份与高基数请求身份。

下一阶段多节点再做：

1. 每节点 agent 或 KubeRay sidecar/DaemonSet。
2. Loki、Elasticsearch 或其他独立后端与 Grafana 查询。
3. offset 持久化、发送重试、缓冲和采集链路自身监控。
4. 按业务需求增加 W3C trace，而不是把 request ID 政名为 trace ID。

## 对现有设计稿的修正建议

现有 [日志系统改造建议](LOGGING_REDESIGN_PROPOSAL.md) 的事件文件、统一查询和留存方向与现实实践一致。需要明确一项长期方向：`container.log` 不是最终架构中的权威应用事件源。完成事件查询工具后，可单独立项把服务命令恢复到 stdout/stderr，交给 Docker logging driver 轮转，并决定是否还要异步归档一份宿主机启动日志。该迁移会改变当前 ARCHITECTURE.md、AGENTS.md 和 harness 约定，因此不应夹在第一阶段事件文件实现中顺手完成。

最终形态不追求“所有日志进入一个文件”，而是做到：来源边界清晰、采集可覆盖、字段可关联、留存有上限、默认视图安静、底层原文随时可展开。
