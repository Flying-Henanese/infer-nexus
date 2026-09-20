#!/usr/bin/env python3
"""Query infer-nexus application events and node-local Ray diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys


# Running ``python scripts/logs.py`` should work from a source checkout without
# requiring callers to set PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infer_nexus.observability.log_query import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
