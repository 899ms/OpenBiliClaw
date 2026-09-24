"""Socratic dialogue module.

Handles deep, probing conversations with the user to better understand them.
The dialogue style is inspired by the Socratic method:
- Ask "why" to uncover motivations
- Propose hypotheses and test them
- Confirm understanding before adjusting
- Adapt dynamically based on responses
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from datetime import tzinfo

    from openbiliclaw.agent.loop import AgentEvent, AgentLoop
    from openbiliclaw.agent.skill import SkillDefinition
    from openbiliclaw.agent.tools import ToolRegistry
    from openbiliclaw.llm.service import LLMService, ModuleOverride, SupportsComplete
    from openbiliclaw.soul.dialogue_learn_queue import DialogueSettlementQueue
    from openbiliclaw.soul.dialogue_turn_context import DialogueTurnBinding
    from openbiliclaw.soul.engine import SoulEngine

logger = logging.getLogger(__name__)

# Cap the dialogue history folded into each prompt. Calibration (2026-07-17,
# first-round — revisit after a provider swap): one exchange ≈ 2 short messages
# ≈ 80 tokens; 20 exchanges ≈ 1.6k tokens of history, which keeps the socratic
# prompt bounded without losing the near-term thread. Below the window the
# prompt bytes are unchanged (baseline test), so provider prompt cache still
# fires for short sessions.
DIALOGUE_WINDOW_TURNS = 20


class DialogueLearningMode(StrEnum):
    """Explicit ownership mode for learning after an interactive reply."""

    QUEUED = "queued"
    REPLY_ONLY_TEST = "reply_only_test"
    LEGACY_DIRECT = "legacy_direct"


class DialogueLearningConfigurationError(RuntimeError):
    """Raised when queued dialogue learning has no settlement queue."""


def _default_turn_timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def format_dialogue_turn_timestamp(
    timestamp: str,
    *,
    local_timezone: tzinfo,
) -> str:
    """Render a recorded turn timestamp without consulting the current clock.

    SQLite ``CURRENT_TIMESTAMP`` values are unmarked UTC. In-memory turns are
    recorded with an explicit local offset. Both pass through this single,
    injectable conversion point before becoming stable prompt bytes.
    """
    normalized = timestamp.strip().replace("Z", "+00:00")
    if not normalized:
        return ""
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        logger.warning("Ignoring invalid dialogue turn timestamp %r", timestamp)
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(local_timezone)
    return f"[{local:%m-%d %H:%M}]"


def _relation_prefix_from_payload(payload: object) -> str:
    """Extract the readable relation prefix from a server-owned turn payload."""
    if not isinstance(payload, Mapping):
        return ""
    raw_binding = payload.get("dialogue_binding")
    if not isinstance(raw_binding, Mapping) or str(raw_binding.get("mode", "")) != "bound":
        return ""
    raw_context = raw_binding.get("context")
    if not isinstance(raw_context, Mapping):
        return ""
    title = str(raw_context.get("title", "")).strip()
    source_type = str(raw_context.get("source_type", "")).strip()
    if not title or source_type not in {"card", "question"}:
        return ""
    label = "卡片" if source_type == "card" else "疑惑问题"
    return f"[回复{label}「{title}」]"


@dataclass
class DialogueTurn:
    """A single turn in a dialogue."""

    role: str  # "user" | "agent"
    content: str
    timestamp: str = field(default_factory=_default_turn_timestamp)
    extracted_insights: list[str] | None = None
    # A durable relation is rendered only for LLM history; ``content`` stays
    # the original user text for UI and audit.
    relation_prefix: str = ""


class SocraticDialogue:
    """Manages Socratic-style dialogue with the user.

    The dialogue module doesn't just record what the user says — it actively
    probes deeper to understand motivations, validate hypotheses, and refine
    the agent's understanding of who the user really is.

    Dialogue strategies:
    1. 追问 Why — Don't stop at preferences, dig into motivations
    2. 提出假设 — Actively hypothesize based on current understanding
    3. 确认验证 — Use recommendations to test hypotheses
    4. 动态调整 — Refine the soul profile based on dialogue
    """

    def __init__(
        self,
        llm: SupportsComplete | None,
        soul_engine: SoulEngine,
        llm_service: LLMService | None = None,
        session: str = "cli",
        tools: list[dict[str, Any]] | None = None,
        tool_dispatcher: Any | None = None,
        module_overrides: Mapping[str, ModuleOverride] | None = None,
        database: Any | None = None,
        local_timezone: tzinfo | None = None,
        now_provider: Callable[[], datetime] | None = None,
        *,
        learning_mode: DialogueLearningMode | str,
        settlement_queue: DialogueSettlementQueue | None = None,
    ) -> None:
        self._llm = llm
        self._soul_engine = soul_engine
        self._llm_service = llm_service
        self._session = session
        self._history: list[DialogueTurn] = []
        default_timezone = datetime.now().astimezone().tzinfo
        self._local_timezone = local_timezone or default_timezone or UTC
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())
        # Phase 1 durable-history regurgitation: after a restart the in-process
        # history is empty, but the durable popup ``chat_turns`` table holds the
        # completed exchanges. Lazily reload them once so a popup session keeps
        # its thread across restarts. Completed chat/hypothesis/confusion rows
        # from every UI session qualify; session remains display ownership only
        # (CLI has no DB and probe scopes remain excluded).
        self._database = database
        self._history_loaded = False
        self._respond_lock = asyncio.Lock()
        self._tools = tools or []
        self._tool_dispatcher = tool_dispatcher
        self._module_overrides = dict(module_overrides) if module_overrides is not None else None
        self._learning_mode = DialogueLearningMode(learning_mode)
        self._settlement_queue = settlement_queue

    @property
    def learning_mode(self) -> DialogueLearningMode:
        """Return the explicit post-reply learning ownership mode."""
        return self._learning_mode

    async def respond(
        self,
        user_message: str,
        *,
        scope: str = "chat",
        turn_id: str = "",
        session: str = "",
        dialogue_binding: DialogueTurnBinding | Mapping[str, object] | None = None,
        progress: Any = None,
    ) -> str:
        """Generate a Socratic response to a user message.

        The response should:
        - Acknowledge what the user said
        - Probe deeper when appropriate ("为什么？")
        - Propose hypotheses ("我猜你可能...")
        - Confirm understanding ("所以你的意思是...")
        - Feel natural and warm, like a friend talking

        Args:
            user_message: The user's message.
            scope: Chat scope threaded to ``learn_from_dialogue`` — only
                unanchored ``"chat"`` runs inventory settles. Probe settlement
                stays in its durable side effect; confusion settlement belongs
                exclusively to the serialized dialogue-anchor processor.
            turn_id: Durable chat-turn id (idempotency observation key).
            session: UI ownership label for this request. The cognitive history
                remains shared across sessions.

        Returns:
            Agent's response.
        """
        if self._learning_mode is DialogueLearningMode.QUEUED and self._settlement_queue is None:
            raise DialogueLearningConfigurationError(
                "queued dialogue learning requires DialogueSettlementQueue"
            )

        binding: DialogueTurnBinding | None = None
        if dialogue_binding is not None:
            from openbiliclaw.soul.dialogue_turn_context import DialogueTurnBinding

            if isinstance(dialogue_binding, DialogueTurnBinding):
                binding = dialogue_binding
            elif isinstance(dialogue_binding, Mapping):
                binding = DialogueTurnBinding.from_mapping(dialogue_binding)
            else:
                raise TypeError("dialogue_binding must be DialogueTurnBinding or a mapping")

        async with self._respond_lock:
            self._ensure_history_loaded()
            history_length = len(self._history)
            turn_timestamp = self._local_now().isoformat()
            self._history.append(
                DialogueTurn(role="user", content=user_message, timestamp=turn_timestamp)
            )

            try:
                service = self._llm_service or self._build_service()
                prompt_message = (
                    binding.render_user_prompt(user_message)
                    if binding is not None
                    else user_message
                )
                prompt_user_message = self._user_prompt_with_current_time(prompt_message)

                # If tools are configured, try tool-calling path first
                if self._tools and self._tool_dispatcher:
                    reply = await self._respond_with_tools(
                        service, prompt_user_message, progress=progress
                    )
                else:
                    response = await service.complete_socratic_dialogue(
                        user_message=prompt_user_message,
                        history=self._history_to_messages(),
                        caller="soul.dialogue",
                    )
                    reply = response.content
            except BaseException:
                del self._history[history_length:]
                logger.exception("Failed to generate Socratic dialogue response.")
                raise

            self._history.append(
                DialogueTurn(
                    role="agent",
                    content=reply,
                    timestamp=self._local_now().isoformat(),
                )
            )
            payload: dict[str, object] = {
                "user_message": user_message,
                "assistant_reply": reply,
                "session": session.strip() or self._session,
                "scope": scope,
                "turn_id": turn_id,
            }
            if binding is not None:
                payload["dialogue_binding"] = binding.to_mapping()
            self._queue_dialogue_learning(payload, binding=binding)
            return reply

    def _queue_dialogue_learning(
        self,
        payload: dict[str, object],
        *,
        binding: DialogueTurnBinding | None = None,
    ) -> None:
        """Submit post-reply learning according to the learning mode."""
        if self._learning_mode is DialogueLearningMode.QUEUED:
            from openbiliclaw.soul.dialogue_learn_queue import (
                ANCHOR_NOT_APPLICABLE,
                AnchorAdmissionSnapshot,
                AnchorPersisted,
                DialogueJobKind,
            )

            queue = self._settlement_queue
            assert queue is not None
            # ``submit`` synchronously freezes the queue-global logical
            # anchor head before the immutable learn envelope is put.
            frozen_snapshot: AnchorAdmissionSnapshot | None = None
            if binding is not None:
                if binding.mode.value == "bound" and binding.context is not None:
                    frozen_snapshot = AnchorPersisted(
                        kind=binding.context.kind,
                        ref=binding.context.ref,
                        generation=binding.context.generation,
                    )
                else:
                    frozen_snapshot = ANCHOR_NOT_APPLICABLE
            if frozen_snapshot is None:
                # Keep the long-standing queue protocol for ordinary
                # unbound turns.  Besides avoiding an unnecessary marker,
                # this keeps lightweight queue adapters source-compatible.
                admitted = queue.submit(DialogueJobKind.LEARN, payload)
            else:
                admitted = queue.submit(
                    DialogueJobKind.LEARN,
                    payload,
                    _server_frozen_anchor_snapshot=frozen_snapshot,
                )
            if admitted is None:
                raise DialogueLearningConfigurationError(
                    "dialogue settlement queue is not accepting learn jobs"
                )
        elif self._learning_mode is DialogueLearningMode.LEGACY_DIRECT:
            learn_fn = getattr(self._soul_engine, "learn_from_dialogue", None)
            if callable(learn_fn):

                async def _background_learn() -> None:
                    try:
                        # This explicitly named compatibility path is owned
                        # only by CLI/OpenClaw. It preserves their baseline
                        # detached learning semantics without joining the
                        # API settlement queue or worker guard.
                        from openbiliclaw.llm.service import _background_admission_bypass

                        with _background_admission_bypass():
                            await learn_fn(**payload)
                    except Exception:
                        logger.exception("Failed to learn from dialogue turn.")

                asyncio.create_task(_background_learn())

    async def stream_agent_reply(
        self,
        agent_loop: AgentLoop,
        user_message: str,
        *,
        session: str = "",
        scope: str = "chat",
        turn_id: str = "",
        session_id: str = "",
        skill: SkillDefinition | None = None,
        tools: ToolRegistry | None = None,
        skill_switch_guide: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Run the multi-hop agent loop for one chat turn, streaming events.

        Shares the legacy single-hop path's history, persona prompt and
        post-reply learning: the user turn is appended before the loop runs
        (rolled back on failure), the socratic system prompt becomes the
        loop's system instruction, and the completed exchange is recorded
        and queued for learning exactly like ``respond``. The dialogue lock
        is held for the whole run so the shared history stays serialized
        with the legacy path.

        With ``skill`` (M4), the skill's persona prompt and the
        ``skill_switch_guide`` block are layered on top of the base socratic
        system prompt, and ``tools`` (the skill's whitelist subset plus meta
        tools) overrides the loop's registry for this run.

        M7: ``session`` / ``session_id`` / ``turn_id`` are forwarded as the
        loop's approval context so parked hard_write approvals can be traced
        back to the conversation that requested them.
        """
        if self._learning_mode is DialogueLearningMode.QUEUED and self._settlement_queue is None:
            raise DialogueLearningConfigurationError(
                "queued dialogue learning requires DialogueSettlementQueue"
            )

        from openbiliclaw.llm.prompts import build_socratic_dialogue_prompt

        async with self._respond_lock:
            self._ensure_history_loaded()
            history_length = len(self._history)
            self._history.append(
                DialogueTurn(
                    role="user", content=user_message, timestamp=self._local_now().isoformat()
                )
            )
            try:
                service = self._llm_service or self._build_service()
                prompt_user_message = self._user_prompt_with_current_time(user_message)
                tone_profile = None
                build_tone = getattr(service, "_build_dialogue_tone_profile", None)
                if callable(build_tone):
                    tone_profile = build_tone()
                prompt_messages = build_socratic_dialogue_prompt(
                    user_message=prompt_user_message,
                    history=self._history_to_messages(),
                    core_memory_text="",
                    tone_profile=tone_profile,
                    reply_style=str(getattr(service, "reply_style", "") or ""),
                    dialogue_tone_prompt=str(getattr(service, "dialogue_tone_prompt", "") or ""),
                )
                system = prompt_messages[0]["content"] if prompt_messages else ""
                if skill is not None:
                    system = _layer_skill_system_prompt(
                        system, skill, skill_switch_guide=skill_switch_guide
                    )
                system = (
                    f"{system}\n\n{_AGENT_LOOP_GROUND_RULES}"
                    if system
                    else (_AGENT_LOOP_GROUND_RULES)
                )
                reply = ""
                async for event in agent_loop.run(
                    system_instruction=system,
                    user_message=prompt_user_message,
                    history=self._history_to_messages(),
                    tools=tools,
                    approval_context={
                        "session": session.strip() or self._session,
                        "session_id": session_id.strip(),
                        "turn_id": turn_id,
                    },
                ):
                    if event.type == "final":
                        reply = event.text
                    yield event
                if not reply.strip():
                    from openbiliclaw.llm.service import LLMResponseContentError

                    raise LLMResponseContentError("LLM returned an empty response")
            except BaseException:
                del self._history[history_length:]
                logger.exception("Failed to generate agent dialogue response.")
                raise

            self._history.append(
                DialogueTurn(
                    role="agent",
                    content=reply,
                    timestamp=self._local_now().isoformat(),
                )
            )
            self._queue_dialogue_learning(
                {
                    "user_message": user_message,
                    "assistant_reply": reply,
                    "session": session.strip() or self._session,
                    "scope": scope,
                    "turn_id": turn_id,
                }
            )

    async def _respond_with_tools(
        self, service: Any, user_message: str, progress: Any = None
    ) -> str:
        """Attempt a tool-calling response, falling back to normal dialogue.

        The flow:
        1. Ask LLM with tool definitions — it may return a tool_call or text.
        2. If tool_call: execute via dispatcher, feed result back, get final reply.
        3. If text: return as-is.
        """
        from openbiliclaw.llm.prompts import build_socratic_dialogue_prompt

        # Core memory is injected downstream by the service itself
        # (``complete_with_tools`` → ``complete_with_core_memory``), not here.
        # ``core_memory_text`` stays a documented test seam on the builder; in
        # production it is always "".
        core_memory = ""
        tone_profile = None
        build_tone = getattr(service, "_build_dialogue_tone_profile", None)
        if callable(build_tone):
            tone_profile = build_tone()
        prompt_messages = build_socratic_dialogue_prompt(
            user_message=user_message,
            history=self._history_to_messages(),
            core_memory_text=core_memory,
            tone_profile=tone_profile,
            reply_style=str(getattr(service, "reply_style", "") or ""),
            dialogue_tone_prompt=str(getattr(service, "dialogue_tone_prompt", "") or ""),
        )
        system = prompt_messages[0]["content"] if prompt_messages else ""

        response = await service.complete_with_tools(
            system_instruction=system,
            user_input=user_message,
            tools=self._tools,
            history=self._history_to_messages(),
            caller="soul.dialogue.tools",
            bypass_semaphore=True,
        )

        # If the LLM returned a tool call, execute and continue
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            logger.info("Dialogue tool call: %s", tool_call.get("name"))
            if progress is not None:
                await progress(
                    "tool_call",
                    {
                        "name": str(tool_call.get("name", "")),
                        "arguments": tool_call.get("arguments", {}),
                    },
                )
            if self._tool_dispatcher is None:
                return str(response.content)
            tool_result = self._tool_dispatcher.dispatch(tool_call)

            # Feed tool result back to get a natural reply
            followup = await service.complete_socratic_dialogue(
                user_message=f"[工具执行结果] {tool_result}",
                history=self._history_to_messages()
                + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"（调用了工具 {tool_call.get('name')}）"},
                ],
                caller="soul.dialogue.tool_followup",
            )
            return str(followup.content)

        return str(response.content)

    async def extract_insights(self, turns: list[DialogueTurn]) -> list[dict[str, Any]]:
        """Extract insights about the user from dialogue turns.

        Args:
            turns: Recent dialogue turns to analyze.

        Returns:
            List of extracted insight dicts.
        """
        # TODO: Use LLM to identify preference signals, motivations,
        #       personality traits from the conversation
        return []

    @property
    def history(self) -> list[DialogueTurn]:
        """The dialogue history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear the dialogue history."""
        self._history.clear()

    def _ensure_history_loaded(self) -> None:
        """Regurgitate the one durable cognition history across UI sessions."""
        if self._history_loaded:
            return
        self._history_loaded = True
        if self._session == "cli" or self._database is None or self._history:
            return
        try:
            history_lister = getattr(self._database, "list_dialogue_history", None)
            if callable(history_lister):
                rows = history_lister(
                    scopes=("chat", "hypothesis", "confusion"),
                    limit=DIALOGUE_WINDOW_TURNS,
                )
            else:
                lister = getattr(self._database, "list_chat_turns", None)
                if not callable(lister):
                    return
                rows = lister(session="popup", scope="chat", limit=DIALOGUE_WINDOW_TURNS)
        except Exception:
            logger.debug("Failed to regurgitate durable chat history", exc_info=True)
            return
        for row in rows:
            if str(row.get("status", "")) != "completed":
                continue
            scope = str(row.get("scope", "chat")).strip()
            payload = row.get("payload", {})
            if scope == "hypothesis" and isinstance(payload, dict):
                title = str(payload.get("title", "") or row.get("subject_title", "")).strip()
                if title:
                    self._history.append(
                        DialogueTurn(
                            role="agent",
                            content=title,
                            timestamp=str(row.get("created_at", "") or ""),
                        )
                    )
                continue
            if (
                scope == "confusion"
                and isinstance(payload, dict)
                and payload.get("type") == "question"
            ):
                question = str(row.get("reply", "")).strip()
                if question:
                    self._history.append(
                        DialogueTurn(
                            role="agent",
                            content=question,
                            timestamp=str(row.get("created_at", "") or ""),
                        )
                    )
                continue
            message = str(row.get("message", "")).strip()
            reply = str(row.get("reply", "")).strip()
            if not message or not reply:
                continue
            timestamp = str(row.get("created_at", "") or "")
            self._history.append(
                DialogueTurn(
                    role="user",
                    content=message,
                    timestamp=timestamp,
                    relation_prefix=_relation_prefix_from_payload(payload),
                )
            )
            self._history.append(DialogueTurn(role="agent", content=reply, timestamp=timestamp))

    def _history_to_messages(self) -> list[dict[str, str]]:
        """Convert prior dialogue turns to chat messages for the LLM.

        Truncated to the last ``DIALOGUE_WINDOW_TURNS`` exchanges (each ≈ a
        user+agent pair) so the prompt stays bounded. Sessions at or below the
        window are unaffected — the returned bytes match the pre-window
        baseline, keeping provider prompt cache warm for short chats.
        """
        prior = self._history[:-1]
        window_messages = DIALOGUE_WINDOW_TURNS * 2
        if len(prior) > window_messages:
            prior = prior[-window_messages:]
        messages: list[dict[str, str]] = []
        for turn in prior:
            prefix = format_dialogue_turn_timestamp(
                turn.timestamp,
                local_timezone=self._local_timezone,
            )
            relation = f"{turn.relation_prefix} " if turn.relation_prefix else ""
            content = (
                f"{prefix} {relation}{turn.content}" if prefix else f"{relation}{turn.content}"
            )
            messages.append(
                {
                    "role": "assistant" if turn.role == "agent" else turn.role,
                    "content": content,
                }
            )
        return messages

    def _local_now(self) -> datetime:
        current = self._now_provider()
        if current.tzinfo is None:
            return current.replace(tzinfo=self._local_timezone)
        return current.astimezone(self._local_timezone)

    def _user_prompt_with_current_time(self, user_message: str) -> str:
        current = self._local_now()
        raw_offset = current.strftime("%z")
        offset = f"{raw_offset[:3]}:{raw_offset[3:]}" if len(raw_offset) == 5 else raw_offset
        return f"{user_message}\n\n当前时间:{current:%Y-%m-%d %H:%M} {offset}".rstrip()

    def _build_service(self) -> LLMService:
        """Create the shared LLM service when one is not injected."""
        from openbiliclaw.llm.service import LLMService

        shared_service = getattr(self._soul_engine, "_llm_service", None)
        if shared_service is not None:
            return cast("LLMService", shared_service)
        memory = getattr(self._soul_engine, "_memory", None)
        if self._llm is None or memory is None:
            raise RuntimeError("Dialogue service is not configured.")
        module_overrides = self._module_overrides
        if module_overrides is None:
            module_overrides = getattr(self._soul_engine, "_module_overrides", {})
        return LLMService(
            registry=self._llm,
            memory=memory,
            module_overrides=module_overrides or {},
            concurrency=int(getattr(self._soul_engine, "_llm_concurrency", 3)),
            concurrency_gate=getattr(self._soul_engine, "_llm_concurrency_gate", None),
            reply_style=str(getattr(self._soul_engine, "_reply_style", "") or ""),
            dialogue_tone_prompt=str(getattr(self._soul_engine, "_dialogue_tone_prompt", "") or ""),
        )


def _layer_skill_system_prompt(
    base_system: str,
    skill: SkillDefinition,
    *,
    skill_switch_guide: str = "",
) -> str:
    """Layer a chat skill's persona prompt on top of the base socratic prompt.

    The base prompt keeps the shared 阿b persona and tone; the skill block
    narrows the role (人设 + 可用数据/工具入口声明) for this session, and the
    optional switch guide lists the other skills the agent may propose via
    the ``suggest_skill`` meta tool.
    """
    blocks = [
        base_system,
        f"本会话你以「{skill.display_name}」（skill: {skill.name}）的角色工作。"
        f"\n\n{skill.system_prompt}",
    ]
    guide = skill_switch_guide.strip()
    if guide:
        blocks.append(guide)
    return "\n\n".join(block for block in blocks if block.strip())


# Hard ground rules for the multi-hop agent loop, appended to the system
# prompt of every agent turn (all skills). Kept deliberately short: small
# models follow fewer, sharper rules better than long instruction lists.
_AGENT_LOOP_GROUND_RULES = (
    "【工作纪律（最高优先级，必须严格遵守）】\n"
    "1. 工具纪律：需要查询或修改任何数据时，必须实际发起工具调用（tool_call）。"
    "严禁只在回复正文里声称「我来查一下」「已查到」「已改好」而没有真的调用工具，"
    "严禁编造工具调用过程或结果。没有实际调用工具，就不得声称查过或改过。\n"
    "2. 记忆归属：你能读到的记忆、画像和历史来自跨会话共享的记忆底座，"
    "不是本对话独有的。除非当前上下文有明确依据，不要把记忆说成「你在本对话里写的/说的」；"
    "不确定来源时就如实说明不确定，不要断言。\n"
    "3. 会话边界：上下文只包含当前会话（本对话）的近期内容，不是全部历史。"
    "「本对话」「这次聊天」「第一回合」都指当前会话；"
    "不要把当前会话的第一条当成全部历史的最早一条。"
)
