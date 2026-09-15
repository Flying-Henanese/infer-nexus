"""Shared request ID validation and generation."""

from __future__ import annotations

import re
from uuid import uuid4

REQUEST_ID_HEADER = b"x-request-id"
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def validate_request_id(value: str | None) -> str | None:
    """Return a request ID only when it matches the service's safe syntax."""
    if value is None or _REQUEST_ID_PATTERN.fullmatch(value) is None:
        return None
    return value


def new_request_id() -> str:
    """Create the service's canonical generated request ID."""
    return uuid4().hex
