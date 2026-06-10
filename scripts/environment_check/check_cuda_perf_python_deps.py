#!/usr/bin/env python
"""Check Python dependencies used by CUDA profiling and benchmark workflows."""

from __future__ import annotations

import importlib
from importlib import metadata
import json
import shutil
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Check:
    name: str
    kind: str
    purpose: str
    module: str | None = None
    distribution: str | None = None
    probe: Callable[[], tuple[bool, str]] | None = None


@dataclass
class Result:
    name: str
    kind: str
    ok: bool
    detail: str
    purpose: str


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _import_check(check: Check) -> tuple[bool, str]:
    if check.module is None:
        return False, "internal error: no module configured"
    try:
        module = importlib.import_module(check.module)
    except Exception as exc:
        return False, f"import failed: {type(exc).__name__}: {exc}"
    distribution = check.distribution or check.module.split(".", 1)[0].replace("_", "-")
    version = _version(distribution)
    module_file = getattr(module, "__file__", None)
    return True, f"version={version}; module={check.module}; file={module_file or 'built-in'}"


def _torch_cuda_probe() -> tuple[bool, str]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        return False, f"import failed: {type(exc).__name__}: {exc}"

    details = [
        f"torch={getattr(torch, '__version__', 'unknown')}",
        f"torch_cuda_build={getattr(getattr(torch, 'version', None), 'cuda', None)}",
        f"cuda_available={torch.cuda.is_available()}",
        f"device_count={torch.cuda.device_count()}",
    ]
    try:
        details.append(f"cudnn={torch.backends.cudnn.version()}")
    except Exception as exc:
        details.append(f"cudnn_query_failed={type(exc).__name__}: {exc}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            details.append(f"device0={torch.cuda.get_device_name(0)}")
            capability = torch.cuda.get_device_capability(0)
            details.append(f"capability={capability[0]}.{capability[1]}")
        except Exception as exc:
            details.append(f"device_query_failed={type(exc).__name__}: {exc}")
    return torch.cuda.is_available(), "; ".join(details)


def _torch_profiler_probe() -> tuple[bool, str]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        return False, f"import failed: {type(exc).__name__}: {exc}"
    profiler = getattr(torch, "profiler", None)
    if profiler is None:
        return False, f"torch={getattr(torch, '__version__', 'unknown')}; torch.profiler missing"
    activities = getattr(profiler, "ProfilerActivity", None)
    has_cuda_activity = bool(activities is not None and hasattr(activities, "CUDA"))
    return True, (
        f"torch={getattr(torch, '__version__', 'unknown')}; "
        f"torch.profiler=yes; ProfilerActivity.CUDA={has_cuda_activity}"
    )


def _pynvml_probe() -> tuple[bool, str]:
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception as exc:
        return False, f"import failed: {type(exc).__name__}: {exc}"
    try:
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            driver = pynvml.nvmlSystemGetDriverVersion()
            return True, f"driver={driver}; device_count={count}"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        return False, f"NVML query failed: {type(exc).__name__}: {exc}"


def _py_spy_probe() -> tuple[bool, str]:
    path = shutil.which("py-spy")
    if not path:
        return False, "py-spy executable not found in PATH"
    return True, f"executable={path}"


CHECKS: tuple[Check, ...] = (
    Check("httpx", "REQUIRED", "Async OpenAI-compatible benchmark driver", "httpx"),
    Check("numpy", "REQUIRED", "Percentiles and throughput statistics", "numpy"),
    Check("torch cuda", "RECOMMENDED", "CUDA visibility and PyTorch CUDA runtime", probe=_torch_cuda_probe),
    Check("torch.profiler", "RECOMMENDED", "Framework-level CPU/CUDA profiler availability", probe=_torch_profiler_probe),
    Check("nvidia-ml-py", "RECOMMENDED", "Direct NVML GPU utilization, memory, and power sampling", probe=_pynvml_probe),
    Check("prometheus-client", "RECOMMENDED", "Gateway/application Prometheus metrics", "prometheus_client", "prometheus-client"),
    Check("psutil", "RECOMMENDED", "Process CPU/RSS/thread sampling", "psutil"),
    Check("tensorboard", "RECOMMENDED", "Profiler trace visualization", "tensorboard"),
    Check("pandas", "RECOMMENDED", "Benchmark result tables and CSV export", "pandas"),
    Check("pyarrow", "RECOMMENDED", "Parquet storage for benchmark samples", "pyarrow"),
    Check("transformers", "RECOMMENDED", "Tokenizer-based input/output token counting", "transformers"),
    Check("matplotlib", "OPTIONAL", "Static benchmark charts", "matplotlib"),
    Check("plotly", "OPTIONAL", "Interactive benchmark charts", "plotly"),
    Check("aiohttp", "OPTIONAL", "Alternative async load driver", "aiohttp"),
    Check("nvtx", "OPTIONAL", "NVTX ranges for Nsight Systems timeline annotation", "nvtx"),
    Check("cupy", "OPTIONAL", "CUDA array/runtime smoke tests outside PyTorch", "cupy"),
    Check("opentelemetry-api", "OPTIONAL", "Request tracing API", "opentelemetry", "opentelemetry-api"),
    Check("opentelemetry-sdk", "OPTIONAL", "Request tracing SDK", "opentelemetry.sdk", "opentelemetry-sdk"),
    Check("py-spy", "OPTIONAL", "CPU flamegraph executable for Python services", probe=_py_spy_probe),
)


def run_check(check: Check) -> Result:
    if check.probe is not None:
        ok, detail = check.probe()
    else:
        ok, detail = _import_check(check)
    return Result(
        name=check.name,
        kind=check.kind,
        ok=ok,
        detail=detail,
        purpose=check.purpose,
    )


def print_table(results: list[Result]) -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print()
    for result in results:
        status = "OK" if result.ok else ("WARN" if result.kind != "REQUIRED" else "FAIL")
        print(f"[{status:<4}] {result.kind:<11} {result.name}")
        print(f"       purpose: {result.purpose}")
        print(f"       detail : {result.detail}")


def main() -> int:
    json_mode = "--json" in sys.argv
    results = [run_check(check) for check in CHECKS]
    if json_mode:
        print(json.dumps([result.__dict__ for result in results], indent=2, ensure_ascii=False))
    else:
        print_table(results)

    required_missing = [result.name for result in results if result.kind == "REQUIRED" and not result.ok]
    if required_missing:
        print()
        print("Missing required Python dependencies: " + ", ".join(required_missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
