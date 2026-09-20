# 日志系统改造建议

状态：历史讨论稿，已由 `logging-storage-query-redesign-plan.md` 和当前
`docs/ARCHITECTURE.md` supersede。日期：2026-09-18。

本文保留早期取舍证据；其中 `container.log`、wrapper 和“尚未实施”的
描述不是当前运行或运维契约。当前实现使用 Docker stdout/stderr、进程隔离
`events-*.jsonl*` 以及服务级 `ray/` 原始目录。

依据：[现有架构](ARCHITECTURE.md#application-logs-and-request-correlation)、[官方资料调研](LOGGING_MANAGEMENT_RESEARCH_2026-09-18.md)及当前三服务 Compose 实现。

## 目标与范围

保留约定等级、类别和期限内的诊断记录，让日常排错从统一入口开始。完整性不等于记录全部 DEBUG、请求内容或永久保存，也不保证硬崩溃前未写出的记录。

继续使用现有结构化日志、请求 ID、终结事件和脱敏机制。第一版补充存储、查询和留存，不引入集中日志平台，不改变 Ray Serve 部署结构。

## 存储与权威来源

```text
${LOGS_HOST_PATH}/
  ray-head/
    container.log
    events-<role>-<process-instance>.jsonl
    ray/session_.../logs/...
    ray/session_latest -> session_...
  ray-worker/
    container.log
    events-<role>-<process-instance>.jsonl
    ray/...
  serve-deployer/
    container.log
    events-<role>-<process-instance>.jsonl
    ray/...
```

事件文件直接放在实际承载进程的服务目录，不增加 `app/`。示意不表示每个服务一定存在每种角色。物理服务身份与进程角色分别记录：Gateway 的文件跟随实际 placement，不能按角色推断目录。

| 记录类别 | 权威查询来源 | 用途 |
| --- | --- | --- |
| infer-nexus 结构化事件 | `events-*.jsonl` 及其轮转备份 | 请求、拒绝、模型与进程生命周期 |
| Ray Serve 组件记录 | 该节点 session 内 Serve 日志 | 代理、控制器、部署状态 |
| Ray Core、worker、vLLM 记录 | 对应 session 原始文件 | actor 崩溃、引擎与底层故障 |
| 服务命令输出 | `container.log` | 启动、参数校验、部署驱动输出 |

同一事件可能仍出现在 worker stdout 或部署 driver 输出中。应用查询以事件文件为准，不再次计入这些副本；原始模式保留它们及来源。框架文件不一定具备相同 schema，第三方传播日志也不必都处于独立框架文件。

保留 `/tmp/ray` 完整挂载，查询只发现相关日志文件，不递归扫描运行资源、socket 和 metrics 配置。当前 session 只解析一次真实路径，避免与 `session_latest` 重复读取。

## 应用写入

在 `observability/logging.py` 增加只接收项目结构化事件的文件输出，复用现有格式化与脱敏逻辑。现有 `configure_logging/get_logger/bind_log_context` 接口尽量保持稳定。应用事件文件不接收全部 root logger 或第三方自由文本。

每个进程独立文件，由该进程负责轮转；启动时生成唯一 `process_instance`，PID 只用于辅助定位。派生进程必须关闭继承的文件 handler 并重新初始化。重复配置不能增加重复 handler。

沿用现有字段，补充 `schema_version`、物理服务身份、节点身份和进程启动标识。请求和模型字段按事件适用性填写，不要求基础设施记录都有请求 ID。目录与服务身份从实际节点环境读取，避免部署 driver 的 `runtime_env.env_vars` 把自己的服务身份覆盖到远程副本。

Compose 容器内统一使用 `/var/log/infer-nexus`，继续由已有 bind mount 决定宿主机归属；保持非 root 和启动前目录准备。文件位置与 stdout 显示策略分开配置，文件保留完整的约定事件，终端可只显示摘要；暂不全局提高 logger 级别。

日志文件无法写入时向 stderr 明确报告并回退到已有输出入口。持续写入失败应限频告警并让查询入口提示覆盖不完整，不能静默丢弃或制造“日志为空即健康”的结论。

## 查询入口

建议提供 `scripts/logs.py` 作为薄 CLI，文件发现、解析和筛选放在可独立验证的模块内。以下命令是拟议接口，当前尚不存在：

```sh
python scripts/logs.py --since 30m
python scripts/logs.py --follow
python scripts/logs.py --request-id <id>
python scripts/logs.py --model <name> --since 1h
python scripts/logs.py --infra --service ray-head --since 30m
python scripts/logs.py --raw --service ray-worker
```

默认从 `.env` 读取日志根目录，也支持显式路径。默认显示：WARN/ERROR、失败/拒绝 outcome、慢请求、模型与进程关键状态转换，以及可靠识别的基础设施严重故障。预期错误可能是 INFO `request.completed`，不能只按 level 过滤。正常成功请求不默认刷屏，但按请求查询时展示全部关联事件。

默认摘要展示时间、服务/角色、事件、模型、请求 ID、结果或错误码；详细模式展开字段、异常与原始文件位置。按请求查询只能展示实际带关联 ID 的记录，相关底层无 ID 日志需按节点和时间上下文继续定位。

基础设施模式读取 Ray/Serve/vLLM 原始记录，处理多行 traceback；原文模式用于不支持的格式。解析失败与文件不可读要可见，并提供原文及路径。应用模式可控的重复副本直接通过来源选择排除，不凭相似文本合并不同请求的错误。

实时模式处理新进程文件、轮转、session 切换和半写入行。历史模式按时间排序；实时模式允许有限等待后输出，晚到记录明确标识。排序不承诺分布式严格因果顺序。无时间戳文本不能伪装成精确事件时间；保留其文件上下文。

## container.log 与 Docker

第一阶段保留 `run_with_log.sh`，不改为简单 tee，也不让它承担应用事件分类。当前脚本直接重定向，因此 `docker compose logs` 通常没有被包裹命令的输出；日常用统一文件查询入口，Docker 用于查看容器状态和仍进入 Docker 的输出。

长期可以让服务命令恢复 stdout/stderr，由 Docker 负责实时查看和轮转。如仍要求宿主机 `container.log`，需要独立采集/归档链路；这需要修改 ARCHITECTURE.md、AGENTS.md 和 harness 中的现有约定，不能仅删除脚本。第一版不同时引入该迁移，避免增加两个日志改造变量。

## 容量与留存

应用事件由单进程 writer 轮转，退出进程文件另设留存期限和历史总量预算。具体大小、备份数、期限根据实际增长率和副本规模确定。

Ray 已配置 50 MiB 与三份备份，但并非所有文件支持轮转，包括部分 worker 输出和 raylet 日志。不能把该设置解释为整个服务目录的容量上限。需要分别统计：活跃文件、退出进程文件、历史 session、container 输出；活跃 session 的非日志运行文件不能随意删除。

历史 session 清理以确认已停止为前提，不能仅依据不是 `session_latest` 就删除；服务重启及其他 Ray 节点仍可能引用历史运行内容。第一版先提供容量报告与清理预览，再制定可验证的停止确认规则。

当前 `container.log` 由 shell 长期持有文件描述符：外部 rename/create 不保证写入新文件，copytruncate 有丢失窗口。不能把简单 host logrotate 当成严格完整性方案。它暂时保留 append-only 行为；上线留存治理前需决定接受的丢失语义，或单独设计可 reopen 的写入者 / Docker 采集迁移并验证信号、退出码和轮转。这是后续必须闭环的问题。

应用日志同步落盘与缓冲写入的选择通过压测决定。不要默认新增无限队列或静默丢弃策略。磁盘容量异常和日志写入失败应有可见信号。

## 分步落地

1. **确认真实噪声来源。** 对启动失败、请求失败和稳定运行样本统计 logger、事件、文件体积、增长率及转发副本。用户提供的目录大小只说明文件分散，不能证明 Ray head 某一类日志最吵。
2. **事件文件与统一查询。** 增加独立应用 sink、物理服务/进程身份和 CLI。保留当前源头等级、正常成功请求留存与容器输出机制，先通过默认视图降噪。
3. **源头去重与存储治理。** 验证后评估关闭 worker 到 driver 的重复转发；只针对实测无价值高频类别调整。补齐事件轮转、退出进程及 session 留存，并解决 container.log 的轮转语义。
4. **按需求集中采集。** 多服务器或多人历史检索成为实际需求时，采用宿主机采集器接集中平台，继续使用现有 schema。推理进程不直接同步依赖远程日志服务。

## 验收

覆盖同一请求的 Gateway/模型关联、预期拒绝、SSE 失败、模型初始化失败，以及没有应用事件的 actor/Core 崩溃。验证敏感输入不落盘、写入权限失败可见、进程重启无文件冲突、轮转和 session 切换可持续查询、历史查询不重复计数。比较改造前后日志体积、推理吞吐和尾延迟。

本建议只新增设计讨论稿，不修改运行行为；既有未提交计划文件保持原状。
