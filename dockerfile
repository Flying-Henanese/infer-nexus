# 基础镜像：mineru:npu-latest，由docker load 命令导入
# 导入所用的 tar 包来源于 https://atomgit.com/Ascend-SACT/Mineru-Optimization.git 仓库中的镜像mineru-ascend.tar
FROM mineru:npu-latest

# 设置工作目录
WORKDIR /app

# 配置 uv 使用阿里云镜像源，加速依赖安装过程
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir uv -i https://mirrors.aliyun.com/pypi/simple/

# 本地的 pyproject.ascend.toml 复制到容器内并重命名为 pyproject.toml
COPY pyproject.ascend.toml ./pyproject.toml
# 满足UV安装依赖时需要的 README.md 文件
COPY README.md ./README.md

# 使用 uv 安装依赖

RUN uv pip install --system --no-cache .

# Install monitoring/profiling helpers used by Ascend benchmark runs.
# CANN, npu-smi, hccn_tool, msprof, and msopprof are provided by the Ascend
# base image or by host-mounted driver/toolkit paths.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        jq \
        sysstat \
    && for package in \
        prometheus \
        prometheus-node-exporter \
        prometheus-process-exporter \
        grafana \
        grafana-server; do \
        if apt-cache show "${package}" >/dev/null 2>&1; then \
            apt-get install -y --no-install-recommends "${package}"; \
        else \
            echo "Skipping unavailable apt package: ${package}"; \
        fi; \
    done \
    && if [ -f /etc/default/sysstat ]; then \
        sed -i 's/^ENABLED=.*/ENABLED="true"/' /etc/default/sysstat; \
    fi \
    && rm -rf /var/lib/apt/lists/*
