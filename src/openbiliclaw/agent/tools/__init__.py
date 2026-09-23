"""Agent tool registry package (M1: JSON Schema tools + dispatch; M3: v1 tool set)."""

from .common import ToolApprovalRequiredError, ToolComponentUnavailableError
from .context import AgentToolContext, build_agent_tool_registry
from .registry import PermissionLevel, Tool, ToolRegistry, ToolResult, validate_tool_arguments
from .source_tools import build_source_tool_registry

__all__ = [
    "AgentToolContext",
    "PermissionLevel",
    "Tool",
    "ToolApprovalRequiredError",
    "ToolComponentUnavailableError",
    "ToolRegistry",
    "ToolResult",
    "build_agent_tool_registry",
    "build_source_tool_registry",
    "validate_tool_arguments",
]
