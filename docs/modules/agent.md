# 聊天 Agent Loop（聊一聊）

## 概述

`src/openbiliclaw/agent/` 承载「聊一聊」对话的多跳 agent 运行时（设计共识：
`docs/plans/2026-09-23-chat-agent-loop-design.md`）。M1 交付后端核心三件：
JSON Schema 工具注册表、provider 原生 function calling、多跳 `AgentLoop`。
SSE 流式接线（M2）、v1 工具集扩充（M3）、skill 体系（M4）、L2 审批门（M7）
在后续里程碑落地。

## 已实现功能

| 任务 | 状态 | 说明 |
|------|------|------|
| M1 JSON Schema 工具注册表 | ✅ | `agent/tools/registry.py`：`Tool`（name / description / JSON Schema parameters / permission_level / handler）+ `ToolRegistry`（注册、按 skill 白名单 `subset()`、按权限 `filter_by_permission()`、`llm_schemas()` 渲染 OpenAI 格式、`legacy_schemas()` 渲染旧扁平格式、参数校验 + 同步/异步 dispatch） |
| M1 SOURCE_TOOLS 迁移 | ✅ | `agent/tools/source_tools.py` 是 create_source / list_sources / toggle_source 的唯一事实来源（JSON Schema + 权限级：list=read，create/toggle=hard_write）；`sources/tools.py` 保留 `SOURCE_TOOLS` 旧扁平结构与 `SourceToolDispatcher` 同步接口，委托同一组 handler |
| M1 原生 function calling | ✅ | 见 [llm 模块](llm.md)：OpenAI 系 chat-completions flavor 原生 FC，其余 provider 走 prompt 模拟兜底 |
| M1 多跳 AgentLoop | ✅ | `agent/loop.py`：`AgentLoop.run()` 异步生成器逐跳产出事件，默认 64 跳上限（`[agent]` 配置），超限后无工具收尾汇报 |
| M2 SSE 流式接线 | ⬜ | 把 `AgentEvent` 接到 `/api/chat/stream` |
| M3 v1 工具集（10–12 个） | ⬜ | 画像/记忆/推荐/B 站数据/discovery/配置只读等 |
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
    └── source_tools.py  # 订阅源管理三工具的 JSON Schema 定义与 handler
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

### 服务层入口

`AgentLoop` 只依赖 `LLMService.complete_with_native_tools()`（协议见
`SupportsNativeToolCompletion`）：路由到支持原生 FC 的 provider 时走 native
链，否则展平消息走 prompt 模拟。详见 [llm 模块](llm.md)。

## 配置

见 [配置参考](config.md) 的 `[agent]` 段：`loop_max_steps`（默认 64）、
`tool_result_max_chars`（默认 4000）。
