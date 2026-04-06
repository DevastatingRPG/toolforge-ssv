# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic integration test for ToolforgeEnvironment.

This test executes exactly three explicit steps:
1. Propose a valid macro tool
2. Use the approved macro tool
3. Propose an invalid macro tool containing wrong_tool
"""

import logging
from typing import List

from models import MacroProposal, ToolCall, ToolforgeAction
from server.inputs.factory import create_input_provider
from server.inputs.simulated.task_selector import TaskSelector
from server.toolforge_env_environment import ToolforgeEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _banner(title: str) -> None:
    logger.info("=" * 90)
    logger.info(title)
    logger.info("=" * 90)


def _log_plan(label: str, plan: List[ToolCall]) -> None:
    logger.info("%s", label)
    for idx, call in enumerate(plan, start=1):
        logger.info(
            "  %d. tool=%s params=%s",
            idx,
            call.tool_name,
            call.params,
        )


def _log_step_result(step_name: str, obs) -> None:
    logger.info("%s result", step_name)
    logger.info("  reward=%.3f done=%s", obs.reward, obs.done)
    logger.info("  summary=%s", obs.metadata.get("summary"))
    logger.info("  plan_accepted=%s", obs.metadata.get("plan_accepted"))
    logger.info("  progression=%s", obs.metadata.get("progression"))
    logger.info("  macro_decision=%s", obs.metadata.get("macro_decision"))
    logger.info("  macro_reason=%s", obs.metadata.get("macro_reason"))


def test_environment_three_step_macro_flow() -> None:
    """Run three explicit steps to validate macro proposal and usage behavior."""
    _banner("TEST: Three-Step Macro Flow")

    task_selector = TaskSelector(mode="eval")
    env = ToolforgeEnvironment(
        task_selector=task_selector,
        input_provider_factory=create_input_provider,
    )

    obs = env.reset(
        episode_id="test-episode-001",
        mode="eval",
        difficulty="easy",
        seed=42,
    )

    logger.info("Reset successful")
    logger.info("  episode_id=%s", env.state.episode_id)
    logger.info("  current_task=%s", obs.current_task.id)
    logger.info("  task_prompt=%s", obs.current_task.prompt)
    logger.info("  required_slots=%s", obs.current_task.required_slots)
    logger.info("  done=%s", obs.done)
    assert not obs.done, "Episode should not be done right after reset"

    _banner("Step 1: Propose valid macro (should clear validation + slot stage)")
    macro_steps = [
        ToolCall(
            tool_name="deploy",
            params={"service_name": "frontend-web", "version": "v2.1.0"},
        ),
        ToolCall(
            tool_name="healthcheck",
            params={"service_name": "frontend-web"},
        ),
        ToolCall(
            tool_name="notify",
            params={"channel": "#deployments", "message": "frontend-web deployed"},
        ),
    ]
    _log_plan("Step 1 plan", macro_steps)

    action_step_1 = ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=macro_steps,
        macro_proposal=MacroProposal(
            name="deploy_and_verify_macro",
            description="Deploy, healthcheck, and notify",
            steps=[macro_steps[0], macro_steps[1]],
        ),
        reasoning="Propose reusable macro for deployment verification.",
    )

    obs = env.step(action_step_1)
    _log_step_result("Step 1", obs)
    assert obs.metadata.get("macro_decision") == "approved", "Expected valid macro approval"
    assert obs.metadata.get("plan_accepted") is True, "Step 1 should pass validation"

    _banner("Step 2: Use approved macro (should clear validation + slot stage)")
    step_2_plan = [
        ToolCall(
            tool_name="deploy_and_verify_macro",
            params={},
        ),
        ToolCall(
            tool_name="restart",
            params={"service_name": "backend-api"},
        ),
        ToolCall(
            tool_name="healthcheck",
            params={"service_name": "backend-api"},
        ),
        ToolCall(
            tool_name="notify",
            params={"channel": "#backend-ops", "message": "deployment complete"},
        ),
        ToolCall(
            tool_name="healthcheck",
            params={"service_name": "backend-api"},
        ),
    ]
    _log_plan("Step 2 plan", step_2_plan)

    action_step_2 = ToolforgeAction(
        action_type="propose_plan",
        plan=step_2_plan,
        macro_proposal=None,
        reasoning="Use approved macro in the next plan.",
    )

    obs = env.step(action_step_2)
    _log_step_result("Step 2", obs)
    assert obs.metadata.get("plan_accepted") is True, "Expected macro usage plan to pass validation"

    _banner("Step 3: Propose invalid macro with wrong_tool")
    step_3_plan = [
        ToolCall(
            tool_name="deploy",
            params={"service_name": "backend-api", "version": "v1.4.2"},
        )
    ]
    _log_plan("Step 3 plan", step_3_plan)

    action_step_3 = ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=step_3_plan,
        macro_proposal=MacroProposal(
            name="invalid_macro",
            description="Contains unknown tool and should be rejected",
            steps=[
                ToolCall(
                    tool_name="deploy",
                    params={"service_name": "backend-api", "version": "v1.4.2"},
                ),
                ToolCall(
                    tool_name="wrong_tool",
                    params={},
                ),
            ],
        ),
        reasoning="Intentionally include wrong_tool to test rejection path.",
    )

    obs = env.step(action_step_3)
    _log_step_result("Step 3", obs)
    assert obs.metadata.get("macro_decision") == "rejected", "Expected invalid macro rejection"

    _banner("Summary")
    logger.info("Final step_count=%d", env.state.step_count)
    logger.info("Accepted macros=%d", len(env.state.accepted_macros))
    logger.info("Rejected macros=%d", env.state.rejected_macro_count)
    logger.info("Tokens used=%d", env.state.tokens_used)

    assert env.state.step_count == 3, "Expected exactly three executed steps"
    assert any(tool.name == "deploy_and_verify_macro" for tool in env.state.accepted_macros), (
        "Expected approved macro to exist in accepted_macros"
    )


if __name__ == "__main__":
    try:
        test_environment_three_step_macro_flow()
        logger.info("ALL TESTS PASSED")
    except AssertionError as exc:
        logger.error("Test assertion failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("Test failed with exception: %s", exc)
        raise
