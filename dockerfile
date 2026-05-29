# 基础镜像：mineru:npu-latest，由docker load 命令导入
# 导入所用的 tar 包来源于 https://atomgit.com/Ascend-SACT/Mineru-Optimization.git 仓库中的大文件mineru-ascend.tar
FROM mineru:npu-latest

# 设置工作目录
WORKDIR /app

# 配置 uv 使用阿里云镜像源，加速依赖安装过程
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir uv -i https://mirrors.aliyun.com/pypi/simple/

# 本地的 pyproject.ascend.toml 复制到容器内并重命名为 pyproject.toml
COPY pyproject.ascend.toml ./pyproject.toml
COPY README.md ./README.md

# 使用 uv 安装依赖
RUN uv sync --no-cache
