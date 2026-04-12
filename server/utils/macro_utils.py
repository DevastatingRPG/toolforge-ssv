# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional
try:
    from ...models import (
        Tool,
        ToolForgeAction,
        ToolForgeState,
    )
except ImportError:
    from models import (
        Tool,
        ToolForgeAction,
        ToolForgeState,
    )
try:
    from ..tools import AbstractToolStore
except ImportError:
    from server.tools import AbstractToolStore

logger = logging.getLogger(__name__)

def process_macro_proposal(
    action: ToolForgeAction,
    state: ToolForgeState,
    tool_store: AbstractToolStore,
    can_accept: bool,
    reject_reason: str,
) -> Dict[str, Any]:
    """Evaluate and apply macro proposal lifecycle for this step."""

    result: Dict[str, Any] = {
        "attempted": False,
        "decision": "none",
        "name": None,
        "reason": "no_macro_proposal",
    }

    has_macro_intent = (
        action.action_type == "propose_plan_with_macro"
        or action.macro_proposal is not None
    )
    if not has_macro_intent:
        return result

    result["attempted"] = True

    if action.action_type != "propose_plan_with_macro":
        return reject_macro(
            result=result,
            state=state,
            name=action.macro_proposal.name if action.macro_proposal else None,
            reason="macro_proposal_requires_propose_plan_with_macro_action_type",
        )

    if action.macro_proposal is None:
        return reject_macro(
            result=result,
            state=state,
            name=None,
            reason="missing_macro_proposal_payload",
        )

    proposal = action.macro_proposal
    macro_name = proposal.name.strip()

    if not can_accept:
        return reject_macro(
            result=result,
            state=state,
            name=macro_name,
            reason=reject_reason,
        )

    if not macro_name:
        return reject_macro(
            result=result,
            state=state,
            name=None,
            reason="macro_name_cannot_be_empty",
        )

    if not proposal.steps:
        return reject_macro(
            result=result,
            state=state,
            name=None,
            reason="macro_steps_cannot_be_empty",
        )

    if tool_store.get_tool(macro_name) is not None:
        return reject_macro(
            result=result,
            state=state,
            name=macro_name,
            reason="macro_name_already_exists",
        )

    if len(proposal.steps) < 2:
        return reject_macro(
            result=result,
            state=state,
            name=macro_name,
            reason="macro_requires_at_least_two_steps",
        )

    missing_steps = [
        call.tool_name
        for call in proposal.steps
        if tool_store.get_tool(call.tool_name) is None
    ]
    if missing_steps:
        return reject_macro(
            result=result,
            state=state,
            name=macro_name,
            reason=f"macro_contains_unknown_tools:{','.join(missing_steps)}",
        )

    if any(
        (tool := tool_store.get_tool(call.tool_name)) is not None and tool.is_macro
        for call in proposal.steps
    ):
        return reject_macro(
            result=result,
            state=state,
            name=macro_name,
            reason="nested_macro_steps_not_supported",
        )

    composed_of: List[str] = [call.tool_name for call in proposal.steps]

    macro_tool = Tool(
        name=macro_name,
        description=proposal.description.strip() or f"Macro: {' -> '.join(composed_of)}",
        is_macro=True,
        steps=proposal.steps,
    )

    tool_store.add_tool(macro_tool)
    state.accepted_macros.append(macro_tool)
    state.available_tools = tool_store.get_all_tools()

    # Store macro definition for sequence-based recognition tracking
    state.macro_definitions[macro_name] = composed_of

    result["decision"] = "approved"
    result["name"] = macro_name
    result["reason"] = "macro_registered"
    logger.info(
        "Macro approved: name='%s', steps=%s",
        macro_name,
        composed_of,
    )
    return result

def reject_macro(
    result: Dict[str, Any],
    state: ToolForgeState,
    name: Optional[str],
    reason: str,
) -> Dict[str, Any]:
    """Record macro rejection and return standardized metadata payload."""

    state.rejected_macro_count += 1
    result["decision"] = "rejected"
    result["name"] = name
    result["reason"] = reason
    logger.info("Macro rejected: name='%s', reason='%s'", name, reason)
    return result
