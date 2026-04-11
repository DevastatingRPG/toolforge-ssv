# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List
try:
    from ...models import (
        Tool,
        ToolCall
    )
except ImportError:
    from models import (
        Tool,
        ToolCall
    )

def tool_to_prompt_spec(tool: Tool) -> Dict[str, Any]:
    """Convert a Tool model into a plain prompt-friendly dictionary."""

    return {
        "name": tool.name,
        "description": tool.description,
        "is_macro": tool.is_macro,
        "steps": tool.steps or [],
    }

def available_tools_to_prompt_specs(tools: List[Tool]) -> List[Dict[str, Any]]:
    """Convert available tools to plain dictionaries ready for prompt injection.
    Macros are listed first, followed by atomic tools.
    """
    sorted_tools = sorted(tools, key=lambda t: not t.is_macro)
    return [tool_to_prompt_spec(tool) for tool in sorted_tools]

def analyze_plan(plan: List[ToolCall], available_tools: List[Tool]) -> Dict[str, Any]:
    """Compute deterministic call accounting from server-side tool registry."""

    available_tools_by_name = {
        tool.name: tool for tool in available_tools
    }
    unknown_tool_calls: List[str] = []

    for call in plan:
        tool_def = available_tools_by_name.get(call.tool_name)
        if tool_def is None:
            unknown_tool_calls.append(call.tool_name)

    return {
        "step_call_count": len(plan),
        "unknown_tool_calls": unknown_tool_calls,
    }
