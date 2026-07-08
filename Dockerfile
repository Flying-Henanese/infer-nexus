# syntax=docker/dockerfile:1

ARG RUNTIME_BASE_IMAGE=python:3.11-slim
FROM ${RUNTIME_BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --preview-features extra-build-dependencies --extra serve --extra vllm --extra artifacts --no-dev

COPY config ./config
COPY scripts ./scripts
COPY docs ./docs

EXPOSE 8000 8265
