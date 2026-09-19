# infer-nexus 日志管理调研

日期：2026-09-18。性质：调研与建议，尚未实施，不替代 ARCHITECTURE.md。

## 结论

现有讨论的方向合理：保留框架原始记录，把应用事件结构化，并提供统一排错视图。需要补齐的不是目录层级，而是采集覆盖、唯一权威来源、写入隔离、留存总量和故障可见性。

服务目录直接放 `events-*.jsonl` 可以采用；是否有 `app/` 子目录是项目约定，不是官方标准。第一版适合单机 Compose 的文件与 CLI 方案。集中采集是后续多人、跨节点和历史检索需求增长后的选择，不必现在引入新平台。

## 调研范围与版本

对照 Python、Docker、Ray、vLLM、OpenTelemetry 和 Loki 官方资料，并核对当前仓库源码。`uv.lock` 为 Ray 2.55.1，`pyproject.toml` 允许 `>=2.30,<2.58`；vLLM 依赖范围为 `>=0.18,<0.19`。Ray 尽量使用 2.55.1 标签下资料，vLLM 使用 0.18.0 文档。Collector/Loki 当前文档用于设计参考，不代表仓库已有这些组件。未连接用户测试服务器，无法判断实际最高频日志来源。

## 一般如何分层

1. **生成**：应用记录稳定事件和关联字段；框架保留自己的诊断记录。
2. **采集**：文件或 stdout 作为本地入口；采集端补充主机、容器和文件来源。
3. **保存**：明确每类记录的轮转、容量、期限和删除规则。
4. **查询**：按时间、错误、模型和请求筛选，展开原始记录及上下文。

上述四层是本报告综合官方机制提出的设计分工，不是某一产品强制的标准。OpenTelemetry 提供日志数据模型，Docker 管理容器输出，Ray 管理节点运行日志，Collector 管理文件采集；没有一个组件自动完成全部职责。

## 官方证据及对本项目的意义

### Python：不能让多个进程直接共写一个普通轮转文件

标准库日志支持线程内安全；多进程写同一文件需要额外协调。官方 cookbook 给出 socket 或 queue 汇聚到单一写入者的方案。因此本项目可以按进程独立写文件，也可以引入单一收集写入者；前者更适合第一版。若派生子进程继承文件 handler，需要关闭并重新初始化，不能只给父进程文件名加 PID 就认为已解决。

来源：[Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes)。

### Docker：日志驱动管理的是 stdout/stderr

Docker 官方推荐一般场景使用默认轮转的 `local` 驱动；现有 Compose 的 `json-file` 已显式设置容量，也可以保留。驱动轮转不能管理应用 bind mount 中自行写入的文件。Docker 私有存储应通过 Docker 工具访问，不应作为宿主机查询工具直接读取的目录。

来源：[Configure logging drivers](https://docs.docker.com/engine/logging/configure/)、[Local driver](https://docs.docker.com/engine/logging/drivers/local/)。

Docker 输出交付默认阻塞；非阻塞模式使用有限缓冲，满时丢弃新消息。因此“低开销、永不阻塞、绝不丢失”不能直接保证，应记录所选行为。

来源：[Delivery mode](https://docs.docker.com/engine/logging/configure/#configure-the-delivery-mode-of-log-messages-from-container-to-log-driver)。

项目当前 `run_with_log.sh` 使用直接重定向，容器命令输出进入宿主机 `container.log`，不再直接进入 Docker 输出入口。若希望 `docker compose logs` 也有摘要，需要另行设计输出链路；简单 tee 不能合并分散在 Ray worker 中的记录，还需验证信号和退出码。

### Ray：session 是运行目录，文件轮转不等于目录留存

Ray 为新 session 建目录，并用 `session_latest` 指向最近一次。worker 输出和 Core/系统记录分散在节点文件中，文件布局不保证向后兼容。**2.55.1 官方文档说明部分文件不轮转，包括 raylet 和 Python/Java worker 输出**。当前设置的 50 MiB 与三份备份不能作为全部 Ray 日志或服务目录的容量承诺。

来源：[Ray 2.55.1 logging guide](https://raw.githubusercontent.com/ray-project/ray/ray-2.55.1/doc/source/ray-observability/user-guides/configure-logging.md)。

由此建议：保留现有挂载；排错入口关注 `logs/`，不递归检索 sockets、metrics 或 runtime_resources；对不轮转文件、退出进程文件和历史 session 分别治理。不能在活跃 session 中按大小任意删除文件。`session_latest` 是发现当前 session 的入口，不应与真实 session 路径同时重复读取。

### Ray Serve：已有组件文件，独立事件文件不是必需品

2.55.1 的 Serve LoggingConfig 提供 JSON/TEXT、日志级别、目录和 access log 控制；组件 logger 已创建独立轮转文件。源码中 JSON 配置用于文件 formatter，stderr handler 仍使用文本 formatter，logger 级别共同限制两者。因此不能假设所有输出都是同一种 JSON schema，也不能用提高 logger 级别保留低级别文件记录。

来源：[2.55.1 schema.py](https://github.com/ray-project/ray/blob/ray-2.55.1/python/ray/serve/schema.py)、[2.55.1 logging_utils.py](https://github.com/ray-project/ray/blob/ray-2.55.1/python/ray/serve/_private/logging_utils.py#L307)。

有两种可行路线：采集现有 Serve/worker 文件并识别应用事件，减少额外写入；或独立保存应用事件，换取稳定 schema、文件发现和应用留存策略。后者适合当前希望服务目录直接提供清楚入口的偏好，但增加重复存储和 handler 维护，应测量写入开销。它不替代 Serve 自带 logger，也不代表必须重复保存所有第三方记录。

默认 worker 到 driver 的转发可能引入重复，节点文件权威来源确定后可评估 `log_to_driver=False`。这是去除转发副本，不是关闭节点记录；需要验证部署失败信息仍可见。

来源：[Ray worker forwarding](https://github.com/ray-project/ray/blob/ray-2.55.1/doc/source/ray-observability/user-guides/configure-logging.md#redirecting-worker-logs-to-the-driver)。

### OpenTelemetry：关联依赖字段，目录只是存储组织

标准模型定义事件时间、观察时间、严重度、资源身份、属性和事件名，也提供可选 trace/span 字段。本项目 JSONL 可保持自己的 schema，再做映射；不必为此立即接 SDK。已有 `request_id` 不等于 W3C `TraceId`，未来分布式追踪应分开保存。

来源：[Logs Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/)。

建议增加或核对：`schema_version`、UTC `timestamp`、`level`、`event`、`source`、物理服务/节点、进程角色、进程启动标识、PID、session/replica/deployment、模型、请求 ID、结果、错误码和耗时。保留当前脱敏规则，不能通过第三方自由文本绕过保护。

### 文件采集：需要跟踪轮转和读取进度

Collector Filelog 支持轮转文件跟踪；配置 storage 可跨采集器重启保存读取偏移，但偏移持久化本身不保证下游不丢失。需要一起考虑重试、队列和缓冲。本项目轻量 CLI 也应处理文件新增、轮转、session 切换和半写入行；“解析失败”要可见，不能静默跳过底层错误。

来源：[Filelog Receiver](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/receiver/filelogreceiver/README.md)。

### logrotate：copytruncate 不能保证完整性

手册指出复制和截断之间可能丢日志。当前 `container.log` 是 shell 打开的长期文件描述符，简单 rename/create 后进程可能继续写旧文件；若采用 host logrotate，必须选择和验证重新打开文件或接受 copytruncate 丢失窗口的方案。应用自有文件优先由唯一写入者负责轮转，不同时交给两个轮转系统管理。

来源：[logrotate manual](https://man7.org/linux/man-pages/man8/logrotate.8.html)。

### vLLM：按日志类别降噪，不能盲目全局 WARNING

0.18.0 区分统计日志和请求日志；请求 DEBUG 可能包含 prompt/token 输入。统计开关还可能影响依赖统计开启的功能。本项目嵌入式引擎不是 `vllm serve` CLI，参数需要通过现有 backend adapter 验证；不能直接照搬 CLI 配置。当前 strict backend 已传入 `request_logger=None`。

来源：[vLLM 0.18.0 serve arguments](https://docs.vllm.ai/en/v0.18.0/cli/serve/)。

### 集中查询：请求 ID 不应成为 Loki 索引标签

Loki 官方要求避免无界高基数标签，适合把请求/trace ID、PID 等放在记录内容或 structured metadata 中。本项目若未来接入 Loki，可用有界的 service/environment/source 等作为标签；模型目录当前预注册，也需控制组合数量。

来源：[Loki cardinality](https://grafana.com/docs/loki/latest/get-started/labels/cardinality/)、[Structured metadata](https://grafana.com/docs/loki/latest/get-started/labels/structured-metadata/)。

## 对目前方案的评估

| 原先思路 | 评估 | 补充条件 |
|---|---|---|
| 保留 Ray 原始运行目录 | 合理 | 运行目录不全是日志，不承诺全部自动轮转 |
| 应用 JSONL 直接放服务目录 | 合理 | 是项目组织约定；稳定命名和字段更关键 |
| 每进程独立写入 | 合理 | 处理派生进程与重启身份，避免 PID 重用冲突 |
| 默认只看关键事件 | 合理 | 包含重要 INFO 生命周期和底层故障线索 |
| 查看时过滤而非统一提高级别 | 合理 | 源头明确禁止敏感 payload；无价值高频日志仍可单独治理 |
| 统一合并三个服务 | 合理 | 跨文件时间排序不是严格因果顺序，实时合并可能晚到 |
| 重复事件折叠 | 有条件 | 只折叠显示；不同请求/副本的相似失败不能随意合并 |
| 每文件轮转即可 | 不充分 | 新进程与历史 session 仍累计，需要总量/期限治理 |

## 本项目建议的第一版

保留三服务结构与现有 Ray 挂载。在每个实际承载进程的服务目录增加应用事件文件。物理服务身份应来自所在容器/节点，不能按角色硬编码为 head 或 worker：Gateway 实际 placement 仍由 Ray 决定。

示意文件名：`events-<role>-<process-instance>.jsonl`。`process-instance` 在每次进程启动时唯一；replica/PID 可以作为可读前缀和记录字段，单独 PID 不足以区分跨容器重启。具体命名在实施时确定，避免敏感信息和不安全路径字符。

应用事件 handler 只收项目事件；第三方日志继续走原始路径。明确事件 JSONL 是应用事件的权威查询来源，Ray worker stdout 副本不再次算入应用结果。底层排错模式额外读取 Ray/Serve/vLLM 和 container 输出，保留原始路径和行位置。

默认显示 WARN/ERROR、应用异常 outcome、慢请求和关键生命周期。请求可能以 INFO `request.completed` 记录预期错误，所以筛选不能只依赖 severity。基础设施故障可能没有应用事件：提供明确的 infrastructure/raw 模式，并给默认入口补充能可靠识别的底层故障。文本解析失败时回退原文，不能把空查询结果当成健康证明。

先保持已记录普通成功请求的留存，采样只用于显示。之后如果确需源头采样，记录采样率及覆盖范围，不再称这些记录为逐请求完整历史。完整性定义为：约定等级、事件类别和期限内的诊断记录；不包含 prompt/body，不保证硬崩溃前尚未写出的记录。

容量由三部分组成：活跃进程文件预算 + 已退出进程历史预算 + Ray/container 及历史 session 预算。按实测每日日志量、最大副本数和排查窗口选数值，不把“7 天”“50 MiB”作为行业固定答案。已有非 root 与宿主机目录准备约束继续保留。

## 分阶段交付与验收

1. 先对真实文件统计各来源数量、体积、增长率和重复情况；选取启动失败、请求失败、正常运行三类样本。
2. 增加服务目录内应用事件文件和查看 CLI，覆盖新增进程、轮转、重启和当前/历史 session。
3. 制定留存及故障降级策略，记录写入失败/磁盘满的可见方式；检查日志开销。
4. 多人跨机器检索成为实际需求后，再评估宿主机采集器与集中日志存储。不要让推理进程同步依赖远程日志平台。

验收至少包含：同一请求的 Gateway/模型关联；预期拒绝与 SSE 错误；模型初始化失败；无应用事件的 actor/Core 崩溃；轮转后持续读取；新进程文件发现；历史 session 查询；重启后无重复计数；敏感输入不落盘；写入权限错误可见；压力下日志量及尾延迟比较。

本次仅新增调研文档，没有修改服务、配置、部署或现有未提交计划文件。
