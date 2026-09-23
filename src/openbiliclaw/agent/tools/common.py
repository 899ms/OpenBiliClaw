"""Shared helpers and error types for the M3 standard tool set.

Handlers never let a missing runtime component escape as an unexpected
exception: they raise ``ToolComponentUnavailableError`` (or return a
readable message for user-correctable input), and ``ToolRegistry.dispatch``
maps the raise to a machine-readable ``handler_error`` result.
"""

from __future__ import annotations

import inspect
from typing import Any


class ToolComponentUnavailableError(RuntimeError):
    """Raised when a handler's required runtime component is not wired."""


class ToolApprovalRequiredError(RuntimeError):
    """Raised by hard_write placeholder handlers pending the M7 approval gate."""


def require_component(component: Any, name: str) -> Any:
    """Return ``component`` or raise a machine-diagnosable unavailability error."""
    if component is None:
        raise ToolComponentUnavailableError(f"组件不可用: {name} 未初始化")
    return component


async def maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable; pass plain values through.

    Production components are async while tests often substitute sync fakes;
    this keeps handlers agnostic to either shape.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def clamp_int(value: Any, *, default: int, minimum: int = 1, maximum: int = 100) -> int:
    """Coerce an LLM-supplied number into a bounded int."""
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def truncate_text(text: str, max_chars: int) -> str:
    """Bound tool output size; append an explicit truncation marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…（已截断，共 {len(text)} 字符）"


def short(value: Any, limit: int = 80) -> str:
    """One-line, length-bounded rendering of a row field."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
