# syntax=docker/dockerfile:1

ARG RUNTIME_BASE_IMAGE=python:3.11-slim

FROM ${RUNTIME_BASE_IMAGE} AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --preview-features extra-build-dependencies --extra serve --extra vllm --extra artifacts --no-dev

FROM ${RUNTIME_BASE_IMAGE} AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY docs ./docs
COPY README.md ./README.md

EXPOSE 8000 8265
