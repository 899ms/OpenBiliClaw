import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const popupHtml = readFileSync(resolve("popup", "popup.html"), "utf8");
const popupJs = readFileSync(resolve("popup", "popup.js"), "utf8");
const popupApi = readFileSync(resolve("popup", "popup-api.js"), "utf8");

test("popup loads the shared agent-chat helper before the module script", () => {
  const shared = popupHtml.indexOf('<script src="shared/agent-chat.js"></script>');
  const module = popupHtml.indexOf('<script type="module" src="popup.js"></script>');
  assert.ok(shared > -1, "shared/agent-chat.js script tag missing");
  assert.ok(shared < module, "shared helper must load before popup.js");
});

test("popup chat tab has sub-tabs for conversation, sessions and tasks", () => {
  assert.match(popupHtml, /id="chatSubtabChat"[^>]*data-chat-subtab="chat"/);
  assert.match(popupHtml, /id="chatSubtabSessions"[^>]*data-chat-subtab="sessions"/);
  assert.match(popupHtml, /id="chatSubtabTasks"[^>]*data-chat-subtab="tasks"/);
  assert.match(popupHtml, /id="chatSubpanelSessions"[^>]*hidden/);
  assert.match(popupHtml, /id="chatSubpanelTasks"[^>]*hidden/);
  assert.match(popupHtml, /id="chatSessionNew"/);
  assert.match(popupHtml, /id="chatSessionsList"/);
  assert.match(popupHtml, /id="chatTasksList"/);
  assert.match(popupHtml, /id="chatTaskDetail"[^>]*hidden/);
  assert.match(popupHtml, /id="chatTasksBadge"[^>]*hidden/);
});

test("popup chat keeps the compact agent bar with skill select and approvals toggle", () => {
  assert.match(popupHtml, /id="chatSkillSelect"[^>]*aria-label="对话角色"/);
  assert.match(popupHtml, /id="chatApprovalsToggle"[^>]*aria-controls="chatApprovalsList"[^>]*hidden/);
  assert.match(popupHtml, /id="chatApprovalsList"[^>]*hidden/);
  // The composer and history layout contract from chat-layout.test.ts is intact.
  assert.match(popupHtml, /<form id="chatForm" class="chat-form">/);
});

test("popup api exposes agent stream, sessions, skills, approvals and tasks", () => {
  assert.match(popupApi, /export async function streamAgentChatTurn/);
  assert.match(popupApi, /\/chat\/agent\/stream/);
  assert.match(popupApi, /export async function fetchChatSkills/);
  assert.match(popupApi, /export async function fetchChatSessions/);
  assert.match(popupApi, /export async function createChatSession/);
  assert.match(popupApi, /export async function updateChatSession/);
  assert.match(popupApi, /export async function fetchChatSessionDetail/);
  assert.match(popupApi, /export async function fetchChatApprovals/);
  assert.match(popupApi, /export async function approveChatApproval/);
  assert.match(popupApi, /export async function rejectChatApproval/);
  assert.match(popupApi, /export async function createAgentTask/);
  assert.match(popupApi, /export async function fetchAgentTasks/);
  assert.match(popupApi, /export async function cancelAgentTask/);
});

test("popup drives the agent stream first and falls back to the legacy stream on 503", () => {
  assert.match(popupJs, /globalThis\.OpenBiliClawAgentChat/);
  assert.match(popupJs, /async function popupDriveAgentStream\(/);
  assert.match(popupJs, /agentLoopAvailable = false;/);
  assert.match(popupJs, /Number\(error\?\.status\) === 503/);
  // Scoped delight/probe inline chats stay on the legacy single-hop path.
  assert.match(popupJs, /String\(turn\.scope \|\| "chat"\) === "chat"/);
  // Live runs update a targeted slot between user bubble and reply.
  assert.match(popupJs, /function updatePopupAgentRunDom\(turnId\)/);
  assert.match(popupJs, /dataset\.part = "agent-run";/);
  // History replay renders the persisted event log collapsed.
  assert.match(popupJs, /agentEventsFromTurn\(turn\)/);
  assert.match(popupJs, /collapsed: agentRun\.settled \|\| turn\.status === "completed"/);
});

test("popup wires session switching, approvals, task center and summary cards", () => {
  assert.match(popupJs, /async function switchPopupChatSession\(/);
  assert.match(popupJs, /fetchChatSessionDetail\(popupChatSessionId/);
  assert.match(popupJs, /async function refreshChatApprovals\(/);
  assert.match(popupJs, /renderApprovalCardMarkup\(approval, \{ compact: true \}\)/);
  assert.match(popupJs, /async function refreshAgentTasks\(/);
  assert.match(popupJs, /renderAgentTaskDetailMarkup\(detail, \{ compact: true, markdown: renderMarkdown \}\)/);
  assert.match(popupJs, /isAgentTaskSummaryTurn\(turn\)/);
  assert.match(popupJs, /renderAgentTaskSummaryMarkup/);
  assert.match(popupJs, /data-tasks-back/);
});

test("popup agent styles keep the compact small-window form", () => {
  assert.match(popupHtml, /\.chat-subtabs\s*\{/);
  assert.match(popupHtml, /\.chat-subpanel\s*\{[\s\S]*?overflow:\s*hidden;/);
  assert.match(popupHtml, /\.agent-run\s*\{[\s\S]*?font-size:\s*11px;/);
  assert.match(popupHtml, /\.agent-tool\s*>\s*summary\s*\{[\s\S]*?font-size:\s*11px;/);
  assert.match(popupHtml, /\.chat-approvals-list\s*\{[\s\S]*?max-height:\s*min\(32vh,\s*220px\);/);
  assert.match(popupHtml, /\.agent-task-row\s*\{/);
});
