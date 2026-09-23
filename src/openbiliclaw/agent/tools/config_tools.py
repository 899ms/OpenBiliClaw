"""Config read/write tools (M3).

- ``get_config`` (read): renders the ``Config`` dataclass as JSON with
  recursive redaction — any key containing api_key / cookie / token / secret /
  password / credential is masked before anything reaches the LLM.
- ``update_config`` (hard_write): schema + placeholder only. The handler
  raises ``ToolApprovalRequiredError``; real writes land with the M7
  approval gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from .common import ToolApprovalRequiredError, require_component, truncate_text
from .registry import Tool

if TYPE_CHECKING:
    from .context import AgentToolContext

logger = logging.getLogger(__name__)

_REDACTED = "***已脱敏***"
_SENSITIVE_MARKERS = (
    "api_key",
    "apikey",
    "cookie",
    "token",
    "secret",
    "password",
    "credential",
    "sessdata",
    "access_key",
)
_MAX_OUTPUT_CHARS = 6000


def _redact(value: Any) -> Any:
    """Recursively mask sensitive keys in a config mapping."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if item and any(marker in lowered for marker in _SENSITIVE_MARKERS):
                redacted[key_text] = _REDACTED
            else:
                redacted[key_text] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def build_config_tools(ctx: AgentToolContext) -> list[Tool]:
    """Build the config-domain tools bound to ``ctx``."""
    return [
        Tool(
            name="get_config",
            description=(
                "读取当前系统配置（已脱敏：api_key / cookie / token / 密码等敏感字段"
                "一律打码，绝不返回原文）。可用 section 只看某一段，"
                "如 llm / bilibili / discovery / agent / scheduler。"
            ),
            permission_level="read",
            parameters={
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "配置段名（可空，空则返回全部，超长会截断）",
                    },
                },
                "additionalProperties": False,
            },
            handler=lambda args: _get_config(ctx, args),
        ),
        Tool(
            name="update_config",
            description=(
                "修改系统配置（hard_write：需要用户在对话中逐项审批）。"
                "审批门接入前此工具只登记请求，不会落盘任何修改。"
            ),
            permission_level="hard_write",
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "点分配置路径，如 llm.default_provider",
                    },
                    "value": {
                        "type": "string",
                        "description": "目标值（字符串形式）",
                    },
                    "reason": {"type": "string", "description": "修改原因，可空"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            handler=lambda args: _update_config(ctx, args),
        ),
    ]


def _get_config(ctx: AgentToolContext, args: dict[str, Any]) -> str:
    config = require_component(ctx.config, "config")
    if is_dataclass(config) and not isinstance(config, type):
        data: dict[str, Any] = asdict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        return "组件不可用: config 不是可序列化的配置对象"
    redacted = _redact(data)

    section = str(args.get("section") or "").strip()
    if section:
        if section not in redacted:
            available = ", ".join(sorted(redacted))
            return f"配置中没有「{section}」段。可用段: {available}"
        redacted = {section: redacted[section]}

    payload = json.dumps(redacted, ensure_ascii=False, indent=1, default=str)
    return truncate_text(payload, _MAX_OUTPUT_CHARS)


def _update_config(ctx: AgentToolContext, args: dict[str, Any]) -> str:
    key = str(args.get("key") or "").strip()
    value = str(args.get("value") or "")
    reason = str(args.get("reason") or "").strip()
    logger.info("update_config requested (approval-gated, not applied): %s", key)
    detail = f"{key} = {value}" + (f"（原因: {reason}）" if reason else "")
    raise ToolApprovalRequiredError(
        f"配置修改需要用户在对话中逐项审批；审批门（M7）尚未接入，本次未执行任何写入: {detail}"
    )
