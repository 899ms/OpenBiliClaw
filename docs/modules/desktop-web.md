# 桌面 Web（Desktop Web UI）

## 概述

`src/openbiliclaw/web/desktop/` 是桌面 Web 单页应用（`/web`，入口
`index.html`，主逻辑 `assets/js/app.js` 单文件 IIFE，样式
`assets/css/app.css`）。本文档当前聚焦「聊聊口味」tab 的 agent loop
前端（M8，设计共识见
[聊一聊 Agent Loop 设计](../plans/2026-09-23-chat-agent-loop-design.md)，
后端协议见 [agent 模块](agent.md) 与 [api 模块](api.md)）。

## 已实现功能

| 任务 | 状态 | 说明 |
|------|------|------|
| M8 流式过程展示 | ✅ | `POST /api/chat/agent/stream` 真流式：thinking 过程文本 + 每步一行折叠工具摘要（可展开参数/结果）+ 审批卡内嵌；完成后整体折叠为「过程（N 步）」；`final` 落成答复气泡；历史回放从 `payload.agent_events` 重建同一视图 |
| M8 会话列表 | ✅ | 侧栏（新建/切换/内联改名/归档/显示已归档），默认会话徽标，`active_turns>0` 活跃圆点，标题与预览轮询刷新（复用 2.5s 共享聊天轮询），当前会话持久化到 localStorage |
| M8 skill 切换 | ✅ | 会话条 skill chip（图标+名称）弹出角色选择浮层（`GET /api/chat/skills`），按会话记忆选择、下一回合生效；`suggest_skill` 工具调用渲染为切换卡（一键切换/忽略） |
| M8 审批卡 | ✅ | `approval_request` 渲染审批卡（summary+参数+impact，批准并执行/拒绝可填理由）；侧栏「待审批」入口带未读 badge（轮询 `?status=pending`），抽屉里可批准/拒绝；回放里 `approval_result` 显示审批结局与执行结果 |
| M8 任务中心 | ✅ | 侧栏入口 + 右侧抽屉：任务列表（状态/进度/取消）、详情复用过程流组件渲染 `steps`、完成后 report + 建议清单（逐项确认：soft_write「确认执行」/ hard_write「去对话确认」，v1 统一落成来源会话里的结构化指令消息）；`start_background_task` 确认卡；`agent_task_summary` turn 渲染系统汇总卡 |
| M8 回退与兼容 | ✅ | 探测 `GET /api/chat/skills` 失败 → legacy 模式（布局与行为与 M8 前完全一致）；agent 流 503（`loop_enabled=false`）时当轮回退旧 `/api/chat/stream` 假流式；delight/探针内嵌聊天、假设卡片、待聊确认、对话上下文引用等旧功能不动 |

## 模块结构

```
web/desktop/
├── index.html                       # SPA 骨架；chatPage = 会话侧栏 + 对话区 + 三个浮层
└── assets/
    ├── css/app.css                  # 末尾「聊一聊 Agent Loop（M8）」段：侧栏/过程流/卡片/抽屉样式
    └── js/
        ├── app.js                   # 主逻辑；M8 集中在「agent loop（M8）」注释段
        └── chat-agent-core.js       # M8 纯逻辑层（无 DOM）：SSE 增量解析、
                                     # agent 事件 → 过程视图模型、全部卡片/列表 markup
```

`chat-agent-core.js` 是 classic script，暴露 `globalThis.OpenBiliClawChatAgentCore`
（同时 `module.exports`，供 `tests/js/` 的 node:test 直接引用）。有意不碰 DOM、
不发请求：SSE 解析与过程模型归约可单测，DOM 胶水全部留在 `app.js`。

> 注：移动 Web / 插件 popup（M9）另有共享模块 `web/shared/agent-chat.js`
> 承载同类能力；桌面端的 markup 与桌面布局/样式深度耦合（宽屏侧栏 +
> 抽屉形态），v1 保持独立实现。后续可考虑把 SSE 解析与事件归约两层收敛到
> 共享模块，桌面只保留 markup。

## 关键交互接线（app.js）

- **模式探测**：首次进入聊天 tab 时 `initDesktopAgentChat()` 拉
  `GET /api/chat/skills`；成功进入 agent 模式（`chatPage.has-agent-side`
  两栏布局），失败保持 legacy 单栏布局。
- **发送**：`sendChat()` 在 agent 模式且 `scope=chat` 时转
  `sendAgentChat()`：先 `POST /api/chat/turns`（`streaming=true`，带
  `session_id` 与 `skill`）创建 pending turn，再消费
  `POST /api/chat/agent/stream` 的 SSE；每个事件经
  `OpenBiliClawChatAgentCore.createSseParser` 解析后 apply 进 live 过程模型并
  重渲染。503 时 `legacyStreamForTurn()` 复用旧假流式端点完成同一 turn。
- **历史**：agent 模式下 `refreshDialogueTurns()` 改拉
  `GET /api/chat/sessions/{id}?limit=100`（默认会话收编 legacy turn），
  `selectDialogueTurns` 过滤口径不变；带 `agent_events` 的 turn 由
  `desktopAgentTurnMarkup()` 渲染为「用户气泡 + 折叠过程 + 答复气泡」。
  轮询重渲染时保留过程折叠组件与证据的展开状态（`agentDetailKey`）。
- **事件委托**：`#chatLog` / `#chatApprovalsBody` / `#chatTaskCenterBody`
  统一走 `handleAgentSurfaceClick()`（审批 approve/reject/confirm-reject、
  skill 建议 accept/dismiss、后台任务确认卡、建议清单确认、任务详情打开）；
  批准/拒绝后直接就地更新卡片状态，服务端随后把 `approval_result` 追加进
  turn 回放数据，下一次轮询自动对齐。
- **建议清单执行（v1）**：确认一条建议 = 切到来源会话并发送一条结构化指令
  消息（动作 + 参数 JSON），soft_write 由 agent 当回合直接执行，hard_write
  自然触发审批卡；不在前端直接调写接口。

## 测试

`tests/js/desktop-chat-agent-core.test.mjs`（node:test，24 条）：SSE 分片/
CRLF/多行 data/坏帧容错、过程模型归约（thinking/tool_call/tool_result 配对、
approval_request → approval_result 结局、step_limit、error）、折叠组件 markup
（完成后默认折叠、live 展开、HTML 转义）、特殊工具卡（suggest_skill /
start_background_task）、会话/任务/审批/skill 列表 markup。

运行：`node --test tests/js/*.test.mjs`
