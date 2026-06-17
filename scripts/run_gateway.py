"""Start the infer-nexus HTTP gateway using Uvicorn.

Loads service host/port from settings and allows CLI overrides for local runs.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from infer_nexus.core.config import load_settings


def parse_args() -> argparse.Namespace:
    """Parse gateway startup arguments."""
    parser = argparse.ArgumentParser(description="Run infer-nexus gateway with Uvicorn.")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override gateway host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override gateway port.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn reload mode for local development.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of Uvicorn worker processes for the gateway.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the gateway app entrypoint with resolved host/port."""
    args = parse_args()
    settings = load_settings(args.settings)
    workers = args.workers or settings.service.workers
    if args.reload and workers != 1:
        raise SystemExit("--reload cannot be used with multiple gateway workers.")
    os.environ["INFER_NEXUS_SETTINGS"] = args.settings
    uvicorn.run(
        "infer_nexus.main:app",
        host=args.host or settings.service.host,
        port=args.port or settings.service.port,
        reload=args.reload,
        workers=workers,
    )


if __name__ == "__main__":
    main()
