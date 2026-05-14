from __future__ import annotations

import argparse

import uvicorn

from infer_nexus.core.config import load_settings


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.settings)
    uvicorn.run(
        "infer_nexus.main:app",
        host=args.host or settings.service.host,
        port=args.port or settings.service.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
