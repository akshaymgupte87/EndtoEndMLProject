"""Validation and JSON encoding for the tiny click/view event example."""

from __future__ import annotations

import json
from typing import Any


def validate_event(value: dict[str, Any]) -> dict[str, Any]:
    required = {"user_idx", "item_idx", "event_type", "event_time"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"event must contain {sorted(required)}")
    if any(isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0 for key in ("user_idx", "item_idx", "event_time")):
        raise ValueError("user_idx, item_idx, and event_time must be non-negative integers")
    if value["event_type"] not in {"click", "view"}:
        raise ValueError("event_type must be click or view")
    return {key: value[key] for key in sorted(required)}


def encode_event(value: dict[str, Any]) -> bytes:
    return json.dumps(validate_event(value), separators=(",", ":")).encode("utf-8")


def decode_event(value: bytes | str) -> dict[str, Any]:
    try:
        return validate_event(json.loads(value))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise ValueError("event must be valid JSON") from error
