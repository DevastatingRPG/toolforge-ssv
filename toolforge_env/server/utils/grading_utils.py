# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict
try:
    from ...models import (
        ToolForgeAction,
        ToolForgeState,
    )
except ImportError:
    from models import (
        ToolForgeAction,
        ToolForgeState,
    )

def update_grading_state(
    state: ToolForgeState,
    pipeline_result,
    macro_result: Dict[str, Any],
    action: ToolForgeAction,
) -> None:
    """Accumulate episode-level grading counters from a single step."""
    g = state.grading
    g.episode_steps += 1

    # Validation failures
    if not pipeline_result.passed_validation:
        g.validation_failures += 1

    # Harmful calls
    if pipeline_result.step_harmful:
        g.harmful_plan_count += 1

    # Correct plans (full slot fill + valid)
    if pipeline_result.step_task_complete and pipeline_result.passed_validation:
        g.correct_plan_count += 1

    # Efficiency tracking (only when slot_ratio == 1.0)
    sr = pipeline_result.step_slot_ratio
    if sr is not None and sr >= 1.0:
        g.fully_correct_efficiency_opportunities += 1
        g.sum_efficiency_score += (pipeline_result.step_efficiency_score or 0.0)

    # Macro creation tracking
    if macro_result["attempted"]:
        g.macro_creation_attempts += 1
        if macro_result["decision"] == "approved":
            g.macro_creation_approved += 1
            if pipeline_result.passed_validation:
                g.macro_creation_correct += 1
            g.macro_creation_bonus_total += (pipeline_result.step_macro_creation_bonus or 0.0)
        elif macro_result["decision"] == "rejected":
            g.macro_rejected_count += 1

    # Macro usage tracking
    macro_names = {m.name for m in state.accepted_macros}
    if any(c.tool_name in macro_names for c in action.plan):
        g.macro_usage_attempts += 1
        if sr is not None and sr >= 0.65:
            g.macro_usage_correct += 1

    # Macro miss penalty tracking
    g.macro_miss_penalty_total += (pipeline_result.step_macro_miss_penalty or 0.0)

    state.grading = g
