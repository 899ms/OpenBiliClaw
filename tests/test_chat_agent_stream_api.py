"""API tests for the true-streaming agent chat endpoint (M2).

``POST /api/chat/agent/stream`` consumes ``AgentLoop.run`` and forwards one
SSE event per ``AgentEvent``; durable turns persist the loop's events into
``payload.agent_events`` for history replay. The legacy fake-streaming
``/api/chat/stream`` path must stay untouched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from openbiliclaw.agent.loop import AgentEvent
from openbiliclaw.api.app import create_app
from openbiliclaw.storage.database import Database

if TYPE_CHECKING:
    from pathlib import Path


class FakeAgentDialogue:
    """Dialogue double streaming a scripted sequence of agent events."""

    def __init__(self, script: list[AgentEvent | Exception]) -> None:
        self._script = script
        self.agent_calls: list[dict[str, Any]] = []
        self.legacy_calls: list[str] = []

    async def stream_agent_reply(
        self,
        agent_loop: Any,
        message: str,
        *,
        session: str = "",
        scope: str = "chat",
        turn_id: str = "",
        session_id: str = "",
        skill: Any = None,
        tools: Any = None,
        skill_switch_guide: str = "",
    ) -> Any:
        self.agent_calls.append(
            {
                "agent_loop": agent_loop,
                "message": message,
                "session": session,
                "scope": scope,
                "turn_id": turn_id,
                "session_id": session_id,
                "skill": skill,
                "tools": tools,
                "skill_switch_guide": skill_switch_guide,
            }
        )
        for item in self._script:
            if isinstance(item, Exception):
                raise item
            yield item

    async def respond(self, message: str, **kwargs: Any) -> str:
        del kwargs
        self.legacy_calls.append(message)
        return "legacy 单跳回复"


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "openbiliclaw.db")
    database.initialize()
    return database


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((event, json.loads(data)))
    return events


def _multi_hop_script() -> list[AgentEvent]:
    return [
        AgentEvent(type="thinking", step=1, text="我先看看你的订阅"),
        AgentEvent(
            type="tool_call",
            step=1,
            tool_name="list_sources",
            arguments={},
            summary="list_sources()",
        ),
        AgentEvent(type="tool_result", step=1, tool_name="list_sources", text="当前没有订阅"),
        AgentEvent(type="final", step=2, text="你还没有订阅任何内容源。"),
    ]


def _app(tmp_path: Path, dialogue: FakeAgentDialogue) -> Any:
    database = _database(tmp_path)
    app = create_app(
        memory_manager=object(),
        database=database,
        soul_engine=object(),
        dialogue=dialogue,
    )
    app.state.runtime_context.agent_loop = object()
    return app


def test_agent_stream_multi_hop_events_and_turn_persisted(tmp_path: Path) -> None:
    dialogue = FakeAgentDialogue(_multi_hop_script())
    app = _app(tmp_path, dialogue)

    with TestClient(app) as client:
        created = client.post(
            "/api/chat/turns",
            json={
                "turn_id": "agent-turn-1",
                "session": "desktop",
                "scope": "chat",
                "message": "我订阅了什么？",
                "streaming": True,
            },
        )
        assert created.status_code == 200
        assert created.json()["status"] == "pending"

        response = client.post(
            "/api/chat/agent/stream",
            json={
                "turn_id": "agent-turn-1",
                "session": "desktop",
                "message": "我订阅了什么？",
            },
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event for event, _data in events] == [
        "thinking",
        "tool_call",
        "tool_result",
        "final",
        "done",
    ]
    assert events[1][1]["tool_name"] == "list_sources"
    assert events[1][1]["summary"] == "list_sources()"
    assert events[2][1]["ok"] is True
    assert events[3][1]["text"] == "你还没有订阅任何内容源。"
    done = events[4][1]
    assert done["reply"] == "你还没有订阅任何内容源。"
    assert done["turn_id"] == "agent-turn-1"
    assert done["skill"] == "taste-companion"

    # The loop ran under the dialogue lease with the turn context threaded.
    assert len(dialogue.agent_calls) == 1
    call = dialogue.agent_calls[0]
    assert call["turn_id"] == "agent-turn-1"
    assert call["scope"] == "chat"
    assert call["message"] == "我订阅了什么？"

    # Durable completion + persisted event stream for history replay.
    row = app.state.runtime_context.database.get_chat_turn("agent-turn-1")
    assert row is not None
    assert row["status"] == "completed"
    assert row["reply"] == "你还没有订阅任何内容源。"
    persisted = row["payload"]["agent_events"]
    assert [event["type"] for event in persisted] == [
        "thinking",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert persisted[1]["tool_name"] == "list_sources"


def test_agent_stream_step_limit_reached_then_final(tmp_path: Path) -> None:
    dialogue = FakeAgentDialogue(
        [
            AgentEvent(
                type="step_limit_reached",
                step=64,
                text="已达到本次任务的步数上限（64 跳）。",
            ),
            AgentEvent(type="final", step=64, text="目前进展如下……"),
        ]
    )
    app = _app(tmp_path, dialogue)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/agent/stream",
            json={"turn_id": "", "message": "整理一下我的观看历史"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event for event, _data in events] == ["step_limit_reached", "final", "done"]
    assert events[0][1]["step"] == 64
    assert events[2][1]["reply"] == "目前进展如下……"


def test_agent_stream_llm_error_maps_to_error_event_and_fails_turn(tmp_path: Path) -> None:
    dialogue = FakeAgentDialogue(
        [
            AgentEvent(type="thinking", step=1, text="我先试试"),
            RuntimeError("provider exploded"),
        ]
    )
    app = _app(tmp_path, dialogue)

    with TestClient(app) as client:
        client.post(
            "/api/chat/turns",
            json={
                "turn_id": "agent-turn-err",
                "session": "popup",
                "message": "你好",
                "streaming": True,
            },
        )
        response = client.post(
            "/api/chat/agent/stream",
            json={"turn_id": "agent-turn-err", "message": "你好"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event for event, _data in events] == ["thinking", "error"]
    assert events[1][1]["error"]

    row = app.state.runtime_context.database.get_chat_turn("agent-turn-err")
    assert row["status"] == "failed"
    assert row["error"] == events[1][1]["error"]
    # Partial events up to the failure are persisted for replay/audit.
    persisted = row["payload"]["agent_events"]
    assert [event["type"] for event in persisted] == ["thinking"]


def test_agent_stream_without_turn_id_is_ephemeral(tmp_path: Path) -> None:
    dialogue = FakeAgentDialogue([AgentEvent(type="final", step=1, text="你好呀")])
    app = _app(tmp_path, dialogue)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/agent/stream",
            json={"message": "你好"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event for event, _data in events] == ["final", "done"]
    assert events[1][1]["turn_id"] == ""
    assert dialogue.agent_calls[0]["scope"] == "chat"


def test_agent_stream_disabled_by_config(tmp_path: Path) -> None:
    dialogue = FakeAgentDialogue([AgentEvent(type="final", step=1, text="不应到达")])
    app = _app(tmp_path, dialogue)
    app.state.runtime_context.config = SimpleNamespace(agent=SimpleNamespace(loop_enabled=False))

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/agent/stream",
            json={"message": "你好"},
        )

    assert response.status_code == 503
    assert dialogue.agent_calls == []


def test_agent_stream_unavailable_without_loop(tmp_path: Path) -> None:
    dialogue = FakeAgentDialogue([AgentEvent(type="final", step=1, text="不应到达")])
    app = _app(tmp_path, dialogue)
    app.state.runtime_context.agent_loop = None

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/agent/stream",
            json={"message": "你好"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event for event, _data in events] == ["error"]


def test_legacy_chat_endpoints_unaffected(tmp_path: Path) -> None:
    dialogue = FakeAgentDialogue([])
    app = _app(tmp_path, dialogue)

    with TestClient(app) as client:
        legacy = client.post("/api/chat", json={"message": "你好"})
        assert legacy.status_code == 200
        assert legacy.json() == {"reply": "legacy 单跳回复"}

        legacy_stream = client.post("/api/chat/stream", json={"message": "你好"})
        assert legacy_stream.status_code == 200
        events = _parse_sse(legacy_stream.text)
        assert events[0][0] == "phase"
        assert events[-1] == ("done", {"reply": "legacy 单跳回复"})

    assert dialogue.agent_calls == []
    assert dialogue.legacy_calls == ["你好", "你好"]
