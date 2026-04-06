# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic integration test for ToolforgeEnvironment.

Tests the full episode lifecycle:
1. Create environment with task provider factory
2. Reset with eval mode and easy difficulty
3. Execute step actions through all tasks in the queue
4. Verify state progression and episode termination
"""

import logging
from server.toolforge_env_environment import ToolforgeEnvironment
from server.inputs.factory import create_input_provider
from server.inputs.simulated.task_selector import TaskSelector
from server.tools import build_atomic_tools
from models import ToolforgeAction, ToolCall

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment_reset_and_step():
    """
    Test basic reset → step → step scenario.
    
    Verifies:
    - Environment initializes correctly
    - Reset provides valid observation and task
    - Step processes actions without crashing
    - Step returns correct done/reward signals
    - State advances through tasks
    """
    logger.info("=" * 80)
    logger.info("TEST: Environment Reset and Step")
    logger.info("=" * 80)

    # ───────────────────────────────────────────────────────────────────────
    # 1. Initialize environment with task provider factory
    # ───────────────────────────────────────────────────────────────────────
    task_selector = TaskSelector(mode="eval")
    env = ToolforgeEnvironment(
        task_selector=task_selector,
        input_provider_factory=create_input_provider
    )
    logger.info("✓ Environment created successfully")

    # ───────────────────────────────────────────────────────────────────────
    # 2. Reset environment in eval mode with easy difficulty
    # ───────────────────────────────────────────────────────────────────────
    reset_kwargs = {
        "episode_id": "test-episode-001",
        "mode": "eval",
        "difficulty": "easy",
        "seed": 42,
    }
    obs = env.reset(**reset_kwargs)
    
    logger.info(f"✓ Reset successful")
    logger.info(f"  Episode ID: {env.state.episode_id}")
    logger.info(f"  Initial Task: {obs.current_task.id}")
    logger.info(f"  Task Prompt: {obs.current_task.prompt[:60]}...")
    logger.info(f"  Task Difficulty: {obs.current_task.difficulty}")
    logger.info(f"  Required Slots: {obs.current_task.required_slots}")
    logger.info(f"  Baseline Call Count: {obs.current_task.baseline_call_count}")
    logger.info(f"  Available Tools: {len(obs.available_tools)}")
    logger.info(f"  Done: {obs.done}")
    logger.info(f"  Reward: {obs.reward}")
    
    assert obs is not None, "Reset should return observation"
    assert obs.current_task is not None, "Reset should provide current task"
    assert not obs.done, "Episode should not be done immediately after reset"
    assert len(obs.available_tools) > 0, "Some tools should be available"

    # ───────────────────────────────────────────────────────────────────────
    # 3. Build sample actions and execute steps
    # ───────────────────────────────────────────────────────────────────────
    step_count = 0
    max_steps = 20  # Limit iterations to prevent infinite loops
    
    logger.info("\n" + "=" * 80)
    logger.info("STEPPING THROUGH EPISODE")
    logger.info("=" * 80)
    
    while not obs.done and step_count < max_steps:
        step_count += 1
        current_task_id = obs.current_task.id
        logger.info(f"\n--- Step {step_count} ---")
        logger.info(f"Current Task ID: {current_task_id}")
        
        # Get tool names from available tools
        available_tool_names = [tool.name for tool in obs.available_tools]
        logger.info(f"Available Tools: {available_tool_names}")
        
        # Create a sample action: deploy service with healthcheck and notify
        # This mirrors a typical DevOps workflow pattern
        sample_plan = [
            ToolCall(
                tool_name="deploy",
                params={
                    "service_name": "test-service",
                    "version": "v1.0.0"
                },
                token_cost=10
            ),
            ToolCall(
                tool_name="healthcheck",
                params={
                    "service_name": "test-service"
                },
                token_cost=5
            ),
        ]
        
        action = ToolforgeAction(
            action_type="propose_plan",
            plan=sample_plan,
            macro_proposal=None,
            reasoning="Deploy service and verify health for ease-difficulty task completion"
        )
        
        logger.info(f"Action Plan: {[f'{call.tool_name}' for call in action.plan]}")
        
        # Execute the action
        obs = env.step(action)
        
        logger.info(f"✓ Step executed")
        logger.info(f"  Reward: {obs.reward:.3f}")
        logger.info(f"  Done: {obs.done}")
        logger.info(f"  Step Count in State: {env.state.step_count}")
        logger.info(f"  Tokens Used: {env.state.tokens_used}")
        logger.info(f"  Tasks Completed: {len(env.state.completed_tasks)}")
        logger.info(f"  Remaining Tasks in Queue: {len(env.state.task_queue)}")
        
        if not obs.done and obs.current_task:
            logger.info(f"  Next Task ID: {obs.current_task.id}")
        
        # Metadata from step
        if hasattr(obs, 'metadata') and obs.metadata:
            logger.info(f"  Progression: {obs.metadata.get('progression', 'N/A')}")
            logger.info(f"  Plan Accepted: {obs.metadata.get('plan_accepted', 'N/A')}")
            logger.info(f"  Macro Attempted: {obs.metadata.get('macro_attempted', 'N/A')}")

    # ───────────────────────────────────────────────────────────────────────
    # 4. Verify episode completion
    # ───────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("EPISODE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Steps Executed: {step_count}")
    logger.info(f"Final State Step Count: {env.state.step_count}")
    logger.info(f"Tasks Completed: {len(env.state.completed_tasks)}")
    logger.info(f"Accepted Macros: {len(env.state.accepted_macros)}")
    logger.info(f"Rejected Macros: {env.state.rejected_macro_count}")
    logger.info(f"Total Tokens Used: {env.state.tokens_used}")
    logger.info(f"Episode Done: {env.state.done}")
    logger.info(f"Final Observation Done: {obs.done}")
    
    # List completed tasks
    if env.state.completed_tasks:
        logger.info(f"\nCompleted Tasks:")
        for task in env.state.completed_tasks:
            logger.info(f"  - {task.id} ({task.difficulty})")
    
    # Assertions to verify correctness
    assert obs.done or step_count >= max_steps, "Episode should be done or max steps reached"
    assert env.state.step_count > 0, "Step count should be incremented"
    assert len(env.state.completed_tasks) > 0, "At least one task should be completed"
    assert env.state.tokens_used >= 0, "Token usage should be non-negative"
    
    logger.info("\n✓ All assertions passed!")
    logger.info("=" * 80)


def test_environment_multiple_tasks_easy():
    """
    Test environment through multiple easy-difficulty tasks.
    
    Verifies:
    - Task queue is properly managed
    - Episode progresses through multiple tasks
    - Each task is properly presented and tracked
    """
    logger.info("\n\n" + "=" * 80)
    logger.info("TEST: Multiple Tasks in Easy Mode")
    logger.info("=" * 80)

    task_selector = TaskSelector(mode="eval")
    env = ToolforgeEnvironment(
        task_selector=task_selector,
        input_provider_factory=create_input_provider
    )

    # Reset with easy difficulty
    obs = env.reset(
        episode_id="test-episode-multi",
        mode="eval",
        difficulty="easy",
        seed=42,
    )
    
    logger.info(f"✓ Reset successful for multi-task test")
    logger.info(f"  Initial Task Queue Size: {len(env.state.task_queue)}")
    
    task_ids_seen = set()
    step_count = 0
    max_steps = 50
    
    while not obs.done and step_count < max_steps:
        step_count += 1
        task_id = obs.current_task.id
        task_ids_seen.add(task_id)
        
        logger.info(f"\nStep {step_count}: Task '{task_id}'")
        logger.info(f"  Difficulty: {obs.current_task.difficulty}")
        logger.info(f"  Queue Size: {len(env.state.task_queue)}")
        
        # Create minimal action plan
        action = ToolforgeAction(
            action_type="propose_plan",
            plan=[
                ToolCall(
                    tool_name="deploy",
                    params={"service_name": "test", "version": "v1.0.0"},
                    token_cost=10
                )
            ],
            macro_proposal=None,
            reasoning="Test action"
        )
        
        obs = env.step(action)
    
    logger.info(f"\n✓ Multi-task test completed")
    logger.info(f"  Total Steps: {step_count}")
    logger.info(f"  Unique Tasks Seen: {len(task_ids_seen)}")
    logger.info(f"  Tasks Completed: {len(env.state.completed_tasks)}")
    logger.info(f"  Episode Done: {obs.done}")
    
    assert len(task_ids_seen) > 0, "Should see at least one task"
    assert len(env.state.completed_tasks) > 0, "Should complete at least one task"
    
    logger.info("✓ Multi-task test passed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    # ───────────────────────────────────────────────────────────────────────
    # Run all tests
    # ───────────────────────────────────────────────────────────────────────
    try:
        test_environment_reset_and_step()
        test_environment_multiple_tasks_easy()
        
        logger.info("\n\n" + "🎉 " * 20)
        logger.info("ALL TESTS PASSED!")
        logger.info("🎉 " * 20)
        
    except AssertionError as e:
        logger.error(f"\n❌ Test assertion failed: {e}")
        raise
    except Exception as e:
        logger.error(f"\n❌ Test failed with exception: {e}")
        raise
