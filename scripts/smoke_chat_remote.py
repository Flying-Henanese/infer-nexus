"""Smoke-test an OpenAI-compatible chat endpoint.

This script is intentionally lightweight: it prints one line per request so the
operator can correlate client-side timing with gateway.log on the remote host.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class RequestResult:
    index: int
    status: str
    elapsed_seconds: float
    http_status: int | None = None
    error: str | None = None
    output_preview: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.0.194:8000/v1")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--api-key")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Explain in one sentence what a Ray Serve queue length warning may mean.",
    )
    return parser.parse_args()


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_stream_delta(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


async def _send_non_streaming(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    index: int,
) -> RequestResult:
    start = time.perf_counter()
    try:
        response = await client.post(url, headers=headers, json=payload)
        elapsed = time.perf_counter() - start
        if response.is_error:
            return RequestResult(
                index=index,
                status="error",
                elapsed_seconds=elapsed,
                http_status=response.status_code,
                error=response.text[:500],
            )
        text = _extract_text(response.json())
        return RequestResult(
            index=index,
            status="ok",
            elapsed_seconds=elapsed,
            http_status=response.status_code,
            output_preview=text[:120],
        )
    except httpx.TimeoutException as exc:
        return RequestResult(
            index=index,
            status="timeout",
            elapsed_seconds=time.perf_counter() - start,
            error=str(exc) or exc.__class__.__name__,
        )
    except Exception as exc:
        return RequestResult(
            index=index,
            status="error",
            elapsed_seconds=time.perf_counter() - start,
            error=f"{exc.__class__.__name__}: {exc}",
        )


async def _send_streaming(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    index: int,
) -> RequestResult:
    start = time.perf_counter()
    text_parts: list[str] = []
    try:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.is_error:
                body = await response.aread()
                return RequestResult(
                    index=index,
                    status="error",
                    elapsed_seconds=time.perf_counter() - start,
                    http_status=response.status_code,
                    error=body.decode("utf-8", errors="replace")[:500],
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line.removeprefix("data:").strip()
                if raw == "[DONE]":
                    break
                if not raw:
                    continue
                text_parts.append(_extract_stream_delta(json.loads(raw)))
        return RequestResult(
            index=index,
            status="ok",
            elapsed_seconds=time.perf_counter() - start,
            http_status=200,
            output_preview="".join(text_parts)[:120],
        )
    except httpx.TimeoutException as exc:
        return RequestResult(
            index=index,
            status="timeout",
            elapsed_seconds=time.perf_counter() - start,
            error=str(exc) or exc.__class__.__name__,
        )
    except Exception as exc:
        return RequestResult(
            index=index,
            status="error",
            elapsed_seconds=time.perf_counter() - start,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def _print_result(result: RequestResult) -> None:
    status = result.http_status if result.http_status is not None else "-"
    line = (
        f"[{result.index:03d}] {result.status.upper():7s} "
        f"http={status} elapsed={result.elapsed_seconds:.2f}s"
    )
    if result.error:
        line += f" error={result.error}"
    if result.output_preview:
        line += f" output={result.output_preview!r}"
    print(line, flush=True)


async def run(args: argparse.Namespace) -> None:
    """Run the smoke test."""
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")

    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = _headers(args.api_key)
    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[RequestResult] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def worker(index: int) -> None:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": f"{args.prompt} request_id={index}"}],
                "max_tokens": args.max_tokens,
                "stream": bool(args.stream),
            }
            async with semaphore:
                if args.stream:
                    result = await _send_streaming(
                        client,
                        url=url,
                        headers=headers,
                        payload=payload,
                        index=index,
                    )
                else:
                    result = await _send_non_streaming(
                        client,
                        url=url,
                        headers=headers,
                        payload=payload,
                        index=index,
                    )
                results.append(result)
                _print_result(result)

        started = time.perf_counter()
        print(
            f"Sending {args.num_requests} request(s) to {url} "
            f"model={args.model!r} concurrency={args.concurrency} stream={args.stream}",
            flush=True,
        )
        await asyncio.gather(*(worker(index) for index in range(1, args.num_requests + 1)))

    total_elapsed = time.perf_counter() - started
    ok = sum(1 for result in results if result.status == "ok")
    timeout_count = sum(1 for result in results if result.status == "timeout")
    error = len(results) - ok - timeout_count
    print(
        f"Summary: ok={ok} timeout={timeout_count} error={error} "
        f"total_elapsed={total_elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    """CLI entrypoint."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
