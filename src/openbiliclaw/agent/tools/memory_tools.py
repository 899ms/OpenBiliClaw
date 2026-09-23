"""Memory reading/writing and history retrieval tools (M3).

- ``read_memory``: five-layer memory read (event / preference / awareness /
  insight / soul) plus the ``core`` prompt rendering.
- ``write_memory``: soft write into an ``agent_notes`` namespace inside the
  four non-soul layers; engine-owned keys and the soul layer are untouchable.
- ``search_history``: keyword / time-range retrieval over durable chat turns
  (``chat_turns``) and behavioral events (``events`` table).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .common import clamp_int, maybe_await, require_component, short, truncate_text
from .registry import Tool

if TYPE_CHECKING:
    from .context import AgentToolContext

logger = logging.getLogger(__name__)

MEMORY_LAYERS = ("event", "preference", "awareness", "insight", "soul")
WRITABLE_LAYERS = ("event", "preference", "awareness", "insight")

_AGENT_NOTES_KEY = "agent_notes"
_NOTE_KEY_PATTERN = re.compile(r"^[\w\-一-鿿]{1,64}$")
_MAX_NOTE_VALUE_CHARS = 2000


def build_memory_tools(ctx: AgentToolContext) -> list[Tool]:
    """Build the memory-domain tools bound to ``ctx``."""
    return [
        Tool(
            name="read_memory",
            description=(
                "读取记忆层。layer=core（默认）返回核心记忆摘要（画像+偏好，即每轮对话"
                "注入的那段）；event/preference/awareness/insight/soul 返回对应层的"
                "原始 JSON 数据（超长会截断，可用 max_chars 调整）。"
            ),
            permission_level="read",
            parameters={
                "type": "object",
                "properties": {
                    "layer": {
                        "type": "string",
                        "enum": ["core", *MEMORY_LAYERS],
                        "description": "记忆层名称，默认 core",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "返回内容的最大字符数，默认 3000，上限 8000",
                    },
                },
            },
            handler=lambda args: _read_memory(ctx, args),
        ),
        Tool(
            name="write_memory",
            description=(
                "把对话中获得的一条观察或偏好写入记忆（写入 agent_notes 命名空间，"
                "不会覆盖系统维护的字段）。layer 只允许 event / preference / awareness / "
                "insight；soul 层（人格核心）禁止写入。"
            ),
            permission_level="soft_write",
            parameters={
                "type": "object",
                "properties": {
                    "layer": {
                        "type": "string",
                        "enum": list(WRITABLE_LAYERS),
                        "description": "目标记忆层",
                    },
                    "key": {
                        "type": "string",
                        "description": "命名空间内的键名（字母/数字/下划线/连字符/中文，≤64 字符）",
                    },
                    "value": {
                        "type": "string",
                        "description": "要记住的内容（≤2000 字符）",
                    },
                },
                "required": ["layer", "key", "value"],
                "additionalProperties": False,
            },
            handler=lambda args: _write_memory(ctx, args),
        ),
        Tool(
            name="search_history",
            description=(
                "检索历史记录。source=chat 搜历史对话，event 搜行为事件（点击/反馈等），"
                "all（默认）两者都搜。支持关键词与 ISO 时间范围（start_time/end_time，"
                "如 2026-09-01 或 2026-09-01T10:00:00）。"
            ),
            permission_level="read",
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "关键词，可空"},
                    "source": {
                        "type": "string",
                        "enum": ["chat", "event", "all"],
                        "description": "检索范围，默认 all",
                    },
                    "start_time": {"type": "string", "description": "起始时间（ISO 格式），可空"},
                    "end_time": {"type": "string", "description": "结束时间（ISO 格式），可空"},
                    "limit": {
                        "type": "integer",
                        "description": "每类最多返回条数，默认 10，上限 50",
                    },
                },
                "additionalProperties": False,
            },
            handler=lambda args: _search_history(ctx, args),
        ),
    ]


async def _read_memory(ctx: AgentToolContext, args: dict[str, Any]) -> str:
    memory = require_component(ctx.memory_manager, "memory_manager")
    layer = str(args.get("layer") or "core").strip().lower()
    max_chars = clamp_int(args.get("max_chars"), default=3000, minimum=200, maximum=8000)
    if layer == "core":
        text = await maybe_await(memory.render_core_memory_prompt())
        return truncate_text(text, max_chars)
    if layer not in MEMORY_LAYERS:
        return f"未知记忆层: {layer}（可选: core / {' / '.join(MEMORY_LAYERS)}）"
    data = (await maybe_await(memory.get_layer(layer))).data
    if not data:
        return f"记忆层 {layer} 当前为空。"
    payload = json.dumps(data, ensure_ascii=False, indent=1, default=str)
    return truncate_text(payload, max_chars)


async def _write_memory(ctx: AgentToolContext, args: dict[str, Any]) -> str:
    memory = require_component(ctx.memory_manager, "memory_manager")
    layer_name = str(args.get("layer") or "").strip().lower()
    key = str(args.get("key") or "").strip()
    value = str(args.get("value") or "").strip()

    if layer_name not in WRITABLE_LAYERS:
        return f"记忆层 {layer_name or '(空)'} 不允许写入（可选: {' / '.join(WRITABLE_LAYERS)}）。"
    if not _NOTE_KEY_PATTERN.fullmatch(key):
        return "键名无效：只允许字母/数字/下划线/连字符/中文，长度 1-64。"
    if not value:
        return "写入内容不能为空。"
    if len(value) > _MAX_NOTE_VALUE_CHARS:
        return f"写入内容过长（{len(value)} 字符），上限 {_MAX_NOTE_VALUE_CHARS}。"

    layer_obj = await maybe_await(memory.get_layer(layer_name))
    existing = layer_obj.data.get(_AGENT_NOTES_KEY)
    notes = dict(existing) if isinstance(existing, dict) else {}
    notes[key] = {
        "value": value,
        "updated_at": datetime.now(UTC).isoformat(),
        "source": "chat_agent",
    }
    layer_obj.update(_AGENT_NOTES_KEY, notes)
    save = getattr(layer_obj, "save", None)
    if callable(save):
        await maybe_await(save())
    logger.info("Agent wrote memory note: %s/%s/%s", layer_name, _AGENT_NOTES_KEY, key)
    return f"已写入记忆 {layer_name}/{_AGENT_NOTES_KEY}/{key}。"


def _parse_iso_datetime(raw: Any, field: str) -> tuple[datetime | None, str]:
    """Parse an optional ISO datetime; return (value, error_message)."""
    text = str(raw or "").strip()
    if not text:
        return None, ""
    try:
        return datetime.fromisoformat(text), ""
    except ValueError:
        return None, f"{field} 格式无效: {text}（需要 ISO 格式，如 2026-09-01）"


async def _search_history(ctx: AgentToolContext, args: dict[str, Any]) -> str:
    db = require_component(ctx.database, "database")
    keyword = str(args.get("keyword") or "").strip()
    source = str(args.get("source") or "all").strip().lower()
    limit = clamp_int(args.get("limit"), default=10, maximum=50)

    start, error = _parse_iso_datetime(args.get("start_time"), "start_time")
    if error:
        return error
    end, error = _parse_iso_datetime(args.get("end_time"), "end_time")
    if error:
        return error

    sections: list[str] = []
    if source in ("chat", "all"):
        searcher = getattr(db, "search_chat_turns", None)
        if not callable(searcher):
            return "组件不可用: database 不支持会话检索（search_chat_turns 缺失）"
        turns = await maybe_await(
            searcher(keyword=keyword, start_time=start, end_time=end, limit=limit)
        )
        lines = [
            f"[{short(turn.get('created_at'), 19)}] 用户: {short(turn.get('message'))}"
            + (
                f"\n    助手: {short(turn.get('reply'))}"
                if str(turn.get("reply") or "").strip()
                else ""
            )
            for turn in turns
        ]
        sections.append("历史对话:\n" + ("\n".join(lines) if lines else "（无匹配）"))
    if source in ("event", "all"):
        events = await maybe_await(
            db.query_events(keyword=keyword, start_time=start, end_time=end, limit=limit)
        )
        lines = [
            f"[{short(event.get('created_at'), 19)}] "
            f"({event.get('event_type', '')}/{event.get('source_platform', '')}) "
            f"{short(event.get('title'))}"
            for event in events
        ]
        sections.append("行为事件:\n" + ("\n".join(lines) if lines else "（无匹配）"))
    return "\n\n".join(sections)
