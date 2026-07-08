# syntax=docker/dockerfile:1

ARG BUILDER_BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04
ARG RUNTIME_BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.11
ARG UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG APP_USER=infer-nexus
ARG APP_UID=10001
ARG APP_GID=10001

FROM ${BUILDER_BASE_IMAGE} AS builder

ARG PYTHON_VERSION
ARG UV_INDEX_URL
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_INDEX_URL=${UV_INDEX_URL}

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        cargo \
        git \
        libssl-dev \
        pkg-config \
        rustc \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python${PYTHON_VERSION} -m pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv venv .venv --python ${PYTHON_VERSION} \
    && uv sync --frozen --preview-features extra-build-dependencies --extra serve --extra vllm --extra artifacts --no-dev --no-install-project --python .venv/bin/python

FROM ${RUNTIME_BASE_IMAGE} AS runtime

ARG PYTHON_VERSION
ARG APP_USER
ARG APP_UID
ARG APP_GID
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HOME=/home/${APP_USER} \
    XDG_CACHE_HOME=/home/${APP_USER}/.cache \
    HF_HOME=/home/${APP_USER}/.cache/huggingface \
    VLLM_CACHE_ROOT=/home/${APP_USER}/.cache/vllm \
    MODELSCOPE_CACHE=/home/${APP_USER}/.cache/modelscope \
    RAY_TMPDIR=/tmp/ray \
    PATH=/app/.venv/bin:$PATH

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        cuda-cudart-dev-12-2 \
        cuda-nvrtc-12-2 \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${APP_GID} ${APP_USER} \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --home-dir /home/${APP_USER} --shell /usr/sbin/nologin ${APP_USER} \
    && mkdir -p /app /models /tmp/ray /home/${APP_USER}/.cache/huggingface /home/${APP_USER}/.cache/modelscope /home/${APP_USER}/.cache/vllm \
    && chown -R ${APP_UID}:${APP_GID} /app /models /tmp/ray /home/${APP_USER}

WORKDIR /app

COPY --from=builder --chown=${APP_UID}:${APP_GID} /app/.venv /app/.venv

USER ${APP_USER}

EXPOSE 8000 8265
