# 聊天 Agent Loop（聊一聊）

## 概述

`src/openbiliclaw/agent/` 承载「聊一聊」对话的多跳 agent 运行时（设计共识：
`docs/plans/2026-09-23-chat-agent-loop-design.md`）。M1 交付后端核心三件：
JSON Schema 工具注册表、provider 原生 function calling、多跳 `AgentLoop`；
M2 把 loop 接上了聊天 SSE 端点（真流式）。v1 工具集扩充（M3）、skill 体系
（M4）、L2 审批门（M7）在后续里程碑落地。

## 已实现功能

| 任务 | 状态 | 说明 |
|------|------|------|
| M1 JSON Schema 工具注册表 | ✅ | `agent/tools/registry.py`：`Tool`（name / description / JSON Schema parameters / permission_level / handler）+ `ToolRegistry`（注册、按 skill 白名单 `subset()`、按权限 `filter_by_permission()`、`llm_schemas()` 渲染 OpenAI 格式、`legacy_schemas()` 渲染旧扁平格式、参数校验 + 同步/异步 dispatch） |
| M1 SOURCE_TOOLS 迁移 | ✅ | `agent/tools/source_tools.py` 是 create_source / list_sources / toggle_source 的唯一事实来源（JSON Schema + 权限级：list=read，create/toggle=hard_write）；`sources/tools.py` 保留 `SOURCE_TOOLS` 旧扁平结构与 `SourceToolDispatcher` 同步接口，委托同一组 handler |
| M1 原生 function calling | ✅ | 见 [llm 模块](llm.md)：OpenAI 系 chat-completions flavor 原生 FC，其余 provider 走 prompt 模拟兜底 |
| M1 多跳 AgentLoop | ✅ | `agent/loop.py`：`AgentLoop.run()` 异步生成器逐跳产出事件，默认 64 跳上限（`[agent]` 配置），超限后无工具收尾汇报 |
| M2 SSE 流式接线 | ✅ | 新端点 `POST /api/chat/agent/stream` 真流式转发 `AgentEvent`；`SocraticDialogue.stream_agent_reply()` 复用 persona prompt / 历史 / 学习队列；loop 事件随 turn 落 `payload.agent_events`；旧 `/api/chat` 与 `/api/chat/stream`（假流式）保持共存 |
| M3 v1 工具集（14 个） | ✅ | 见下文「v1 标准工具集」：`AgentToolContext` + `build_agent_tool_registry()` 总装，read / soft_write / hard_write 三级权限，handler 全部防御性降级 |
| M4 skill 加载与切换 | ⬜ | `*/SKILL.md` 目录约定 + 4 个内置 skill |
| M7 L2 审批门 | ⬜ | hard_write 工具的对话内审批卡 |

## 模块结构

```
agent/
├── loop.py              # AgentLoop + AgentEvent（多跳循环与事件模型）
├── orchestrator.py      # 既有空壳编排器（未接 loop）
├── skill.py             # 既有 Skill ABC + SkillRegistry 骨架（M4 填充）
└── tools/
    ├── registry.py      # Tool / ToolResult / ToolRegistry / validate_tool_arguments
    ├── source_tools.py  # 订阅源管理三工具的 JSON Schema 定义与 handler
    ├── common.py        # 共享错误类型（组件缺失/待审批）与输出辅助
    ├── context.py       # AgentToolContext + build_agent_tool_registry（v1 总装）
    ├── profile_tools.py     # get_profile
    ├── memory_tools.py      # read_memory / write_memory / search_history
    ├── recommendation_tools.py  # get_recommendations / query_discovery_pool
    ├── bilibili_tools.py    # get_watch_history（本地数据层）
    ├── feedback_tools.py    # submit_feedback / save_item（soft_write）
    └── config_tools.py      # get_config（脱敏只读）/ update_config（审批占位）
```

## 公开 API

### AgentLoop

```python
from openbiliclaw.agent.loop import AgentLoop, AgentEvent
from openbiliclaw.agent.tools import ToolRegistry

loop = AgentLoop.from_config(llm_service, tool_registry, config)
# 等价于 AgentLoop(llm_service, tool_registry,
#                  max_steps=config.agent.loop_max_steps,
#                  tool_result_max_chars=config.agent.tool_result_max_chars)

async for event in loop.run(
    system_instruction="你是口味伙伴…",
    user_message="帮我看看我订阅了什么",
    history=[{"role": "user", "content": "…"}, ...],
    tools=tool_registry.subset(["list_sources"]),  # 可选：本次运行的白名单子集
):
    print(event.type, event.to_dict())
```

事件类型（`AgentEvent.type`）：

- `thinking`：带工具调用的中间跳里模型输出的文本（`text`）。最终答复**不重复**
  发 thinking——没有工具调用的跳只发 `final`。
- `tool_call`：一次工具调用（`tool_name` / `arguments` / `summary` 一行摘要 / `step`）。
- `tool_result`：工具执行结果（`text` 已按 `tool_result_max_chars` 截断，
  `ok` 区分成功/失败，`truncated` 标记截断）。未知工具名与参数校验失败都以
  `ok=false` 的结果回填给模型，让模型自我纠正。
- `step_limit_reached`：达到 `max_steps` 时发出一次，随后 loop 追加一条收尾指令
  并做一次**无工具**调用，让模型汇报进展。
- `final`：最终回复文本，每个 run 恰好一个（正常收尾或步数上限收尾）。

`step` 从 1 开始编号；LLM 调用失败直接抛给调用方（M2 映射为 SSE error 事件），
工具失败不抛出。`run()` 内部维护 canonical OpenAI 消息列表（assistant 消息带
`tool_calls`，结果用 `role="tool"` + `tool_call_id` 回填），对原生 FC 与 prompt
模拟两条 service 路径透明。

### ToolRegistry

```python
from openbiliclaw.agent.tools import Tool, ToolRegistry, build_source_tool_registry

registry = build_source_tool_registry(database)
registry.register(Tool(
    name="save_note",
    description="把一条观察写入记忆",
    permission_level="soft_write",   # read / soft_write / hard_write
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    handler=lambda args: "已记下",    # 同步或 async 均可
))

registry.llm_schemas()      # OpenAI tools 格式，给原生 FC
registry.legacy_schemas()   # 旧扁平格式（SOURCE_TOOLS 兼容）
registry.subset([...])               # skill 白名单子集
registry.filter_by_permission("soft_write")  # 权限上限过滤

result = await registry.dispatch("save_note", {"text": "..."})
# ToolResult(ok, content, error) — error ∈ "" / unknown_tool /
# invalid_arguments / handler_error / async_handler
result = registry.dispatch_sync("save_note", {...})  # 旧同步调用方
```

参数校验是 JSON Schema 子集（`type` 含联合类型、`properties`、`required`、
`enum`、`items`、`additionalProperties: false`），刻意不做全量实现；handler 内
部仍保留各自的防御性检查。`permission_level` 目前只是元数据 + 过滤能力，
L2 审批门在 M7 接入。

### v1 标准工具集（M3）

`AgentToolContext`（`agent/tools/context.py`）是一个轻量 dataclass，持有
工具所需的运行时组件引用（`database` / `soul_engine` / `memory_manager` /
`recommendation_engine` / `config` / `event_ingress` / `saved_sync_service`，
字段名与 `api/runtime_context.py` 对齐，生产接线在后续里程碑完成）。
`build_agent_tool_registry(ctx)` 总装全部 14 个工具；除源管理三工具
（构造期绑定 database，无 database 时不注册）外，所有工具始终注册，
组件缺失时 handler 抛 `ToolComponentUnavailableError`，由 dispatch 映射为
机器可读的 `handler_error` 结果回填模型。

| 工具 | 权限 | 说明 |
|------|------|------|
| `get_profile` | read | 当前生效画像（洋葱模型，`SoulEngine.get_profile()` ⊕ 用户覆盖，markdown 渲染） |
| `read_memory` | read | 记忆五层读取（core 摘要或 event/preference/awareness/insight/soul 原始 JSON，可截断） |
| `search_history` | read | 历史对话（`chat_turns` 新增 `Database.search_chat_turns()`）+ 行为事件（`query_events`）关键词/时间范围检索 |
| `get_recommendations` | read | 推荐池头部只读预览（`get_pool_candidates` / `get_pool_candidates_for_platform`），不消耗池、不标记已展示 |
| `get_watch_history` | read | 本地内容历史（clicked/shown/removed 投影）与收藏/稍后再看清单，不触发真实抓取 |
| `query_discovery_pool` | read | discovery 候选池库存：可服务数、待处理数、有货平台、可选抽样 |
| `get_config` | read | 配置只读，api_key/cookie/token/password 等键递归打码 |
| `list_sources` | read | 订阅源列表（M1 已有） |
| `write_memory` | soft_write | 写记忆到各层 `agent_notes` 命名空间（event/preference/awareness/insight），不覆盖引擎字段，soul 层禁写 |
| `submit_feedback` | soft_write | 推荐反馈（like/dislike/dismiss/comment），复用 `POST /api/feedback` 同款 durable 事件流入（event_ingress 幂等 + 推荐行投影 + 轻量认知钩子） |
| `save_item` | soft_write | 本地收藏/稍后再看（`SavedSyncService.save_local(auto_sync=False)`，不同步平台账号） |
| `create_source` | hard_write | 创建订阅源（M1 已有） |
| `toggle_source` | hard_write | 订阅源开关（M1 已有） |
| `update_config` | hard_write | 配置修改占位：仅 schema + 登记，handler 抛 `ToolApprovalRequiredError`，真写入待 M7 审批门 |

上下文策略是「不塞数据，给入口」：工具按需查询系统数据，结果全部有
长度上限（截断并标注）。

### 服务层入口

`AgentLoop` 只依赖 `LLMService.complete_with_native_tools()`（协议见
`SupportsNativeToolCompletion`）：路由到支持原生 FC 的 provider 时走 native
链，否则展平消息走 prompt 模拟。详见 [llm 模块](llm.md)。

### SSE 事件协议（M2，`POST /api/chat/agent/stream`）

请求体复用 durable turn 结构（与 `POST /api/chat/turns` 相同的字段：
`message` 必填，`turn_id` / `session` / `scope` 可选）。带 `turn_id` 时完成
该 pending durable turn（`streaming=True` 创建）并把整段事件流落库；不带
`turn_id` 则为临时运行（不落库）。

响应是 `text/event-stream`，每个 `AgentEvent.to_dict()` 一条 SSE event，
**event 名 = `type`**：

| event | data 字段 | 语义 |
|-------|-----------|------|
| `thinking` | `type` / `step` / `text` | 带工具调用的中间跳里模型输出的文本；无工具调用的跳只发 `final`，不重复发 thinking |
| `tool_call` | `type` / `step` / `tool_name` / `arguments` / `summary` | 一次工具调用；`summary` 是一行折叠摘要（如 `list_sources()`） |
| `tool_result` | `type` / `step` / `tool_name` / `text` / `ok` / `truncated` | 工具执行结果（已按 `tool_result_max_chars` 截断）；`ok=false` 表示未知工具 / 参数校验失败 / handler 异常 |
| `step_limit_reached` | `type` / `step` / `text` | 达到步数上限时发一次，**随后必跟一个 `final`**（无工具收尾汇报） |
| `final` | `type` / `step` / `text` | 最终答复，每个 run 恰好一个；发完后流进入收尾 |
| `done` | `reply` / `turn_id` | 终端事件（端点级，非 loop 事件）；`reply` 即 `final.text` |
| `error` | `error` | LLM 异常等失败的唯一事件（安全文案），发出后流结束；带 `turn_id` 时 turn 置为 `failed` |

`step` 从 1 开始编号。turn 落库时关键步骤（含 thinking / tool_call /
tool_result / step_limit_reached / final）以相同 dict 结构写入
`chat_turns.payload.agent_events`（JSON 数组，免迁移），历史回放直接读
`GET /api/chat/turns/{turn_id}` 的 `payload.agent_events`。

### 接线与并发

- `RuntimeContext._rebuild_components()` 在构造 `SocraticDialogue` 后同步构造
  `ctx.agent_loop = AgentLoop.from_config(llm_service,
  build_source_tool_registry(database), config, caller="agent.chat",
  bypass_semaphore=True)`，随热重载原子 swap。
- 端点在 `DialogueExecutionCoordinator` 租约内运行整个 loop（与旧单跳路径
  串行），历史与学习由 `SocraticDialogue.stream_agent_reply()` 在
  `_respond_lock` 下完成：user turn 先 append（失败回滚）、socratic system
  prompt 作为 loop 的 system instruction、完成后 append agent 答复并按
  learning mode 提交学习任务。
- `[agent] loop_enabled = false` 时端点返回 503；旧端点不受影响。

## 配置

见 [配置参考](config.md) 的 `[agent]` 段：`loop_enabled`（默认 true）、
`loop_max_steps`（默认 64）、`tool_result_max_chars`（默认 4000）。
