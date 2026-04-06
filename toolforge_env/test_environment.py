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

from models import MacroProposal, ToolCall, ToolforgeAction
from server.inputs.factory import create_input_provider
from server.inputs.simulated.task_selector import TaskSelector
from server.toolforge_env_environment import ToolforgeEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment_three_step_macro_flow() -> None:
    """Run three explicit steps to validate macro proposal and usage behavior."""
    logger.info("=" * 80)
    logger.info("TEST: Three-Step Macro Flow")
    logger.info("=" * 80)

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

    logger.info("Reset successful: task=%s done=%s", obs.current_task.id, obs.done)
    assert not obs.done, "Episode should not be done right after reset"

    logger.info("\n--- Step 1: Propose valid macro ---")
    macro_steps = [
        ToolCall(
            tool_name="deploy",
            params={"service_name": "frontend-web", "version": "v2.1.0"},
            token_cost=10,
        ),
        ToolCall(
            tool_name="healthcheck",
            params={"service_name": "frontend-web"},
            token_cost=2,
        ),
    ]
    action_step_1 = ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=macro_steps,
        macro_proposal=MacroProposal(
            name="deploy_and_verify_macro",
            description="Deploy then healthcheck",
            steps=macro_steps,
        ),
        reasoning="Propose reusable macro for deployment verification.",
    )

    obs = env.step(action_step_1)
    logger.info("Step 1 obs=%s", obs)
    logger.info("Step 1 reward=%.3f done=%s", obs.reward, obs.done)
    logger.info("Step 1 macro_decision=%s", obs.metadata.get("macro_decision"))
    assert obs.metadata.get("macro_decision") == "approved", "Expected valid macro approval"

    logger.info("\n--- Step 2: Use approved macro ---")
    action_step_2 = ToolforgeAction(
        action_type="propose_plan",
        plan=[
            ToolCall(
                tool_name="deploy_and_verify_macro",
                params={},
                token_cost=10,
            )
        ],
        macro_proposal=None,
        reasoning="Use approved macro in the next plan.",
    )

    obs = env.step(action_step_2)
    logger.info("Step 2 reward=%.3f done=%s", obs.reward, obs.done)
    logger.info("Step 2 plan_accepted=%s", obs.metadata.get("plan_accepted"))
    assert obs.metadata.get("plan_accepted") is True, "Expected macro usage plan to pass validation"

    logger.info("\n--- Step 3: Propose invalid macro with wrong_tool ---")
    action_step_3 = ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=[
            ToolCall(
                tool_name="deploy",
                params={"service_name": "backend-api", "version": "v1.4.2"},
                token_cost=10,
            )
        ],
        macro_proposal=MacroProposal(
            name="invalid_macro",
            description="Contains unknown tool and should be rejected",
            steps=[
                ToolCall(
                    tool_name="deploy",
                    params={"service_name": "backend-api", "version": "v1.4.2"},
                    token_cost=10,
                ),
                ToolCall(
                    tool_name="wrong_tool",
                    params={},
                    token_cost=10,
                ),
            ],
        ),
        reasoning="Intentionally include wrong_tool to test rejection path.",
    )

    obs = env.step(action_step_3)
    logger.info("Step 3 reward=%.3f done=%s", obs.reward, obs.done)
    logger.info("Step 3 macro_decision=%s", obs.metadata.get("macro_decision"))
    logger.info("Step 3 macro_reason=%s", obs.metadata.get("macro_reason"))
    assert obs.metadata.get("macro_decision") == "rejected", "Expected invalid macro rejection"

    logger.info("\nSummary")
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
