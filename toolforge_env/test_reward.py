import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from models import Tool, ToolCall, ToolforgeAction
from server.inputs.factory import create_input_provider
from server.inputs.simulated.task_selector import TaskSelector
from server.toolforge_env_environment import ToolforgeEnvironment

# Check for LLM credentials in the environment
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_ID = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

logger = logging.getLogger(__name__)

def check_llm_status():
    if HF_TOKEN:
        print(f"--- LLM CONFIG DETECTED ---")
        print(f"  Target: {API_BASE}")
        print(f"  Model:  {MODEL_ID}")
        print(f"  Mode:   LIVE LLM EVALUATION")
    else:
        print(f"--- LLM CONFIG MISSING ---")
        print(f"  HF_TOKEN not found in environment.")
        print(f"  Mode:   SIMULATED EVALUATION (FALLBACK)")
    print("-" * 30)


def configure_logging() -> Path:
    """Configure console and timestamped file logging under reward_logs/."""
    log_dir = Path(__file__).parent / "reward_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"reward_test_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    logger.info("Reward logging initialized: %s", log_file_path)
    return log_file_path


def banner(title: str) -> None:
    line = "=" * 100
    logger.info(line)
    logger.info(title)
    logger.info(line)


def safe_dump(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump"):
            payload = obj.model_dump()
        elif isinstance(obj, dict):
            payload = obj
        else:
            payload = str(obj)
        return json.dumps(payload, indent=2, default=str, ensure_ascii=True)
    except Exception as exc:  # pragma: no cover - defensive logging helper
        return f"<unserializable: {exc}> {obj!r}"


def log_observation(obs: Any) -> None:
    logger.info("Observation snapshot:\n%s", safe_dump(obs))


def log_state(env: ToolforgeEnvironment) -> None:
    state = env.state
    snapshot = {
        "episode_id": getattr(state, "episode_id", None),
        "step_count": getattr(state, "step_count", None),
        "current_task_id": getattr(getattr(state, "current_task", None), "id", None),
        "task_queue_size": len(getattr(state, "task_queue", [])),
        "completed_tasks": [task.id for task in getattr(state, "completed_tasks", [])],
        "accepted_macros": [tool.name for tool in getattr(state, "accepted_macros", [])],
        "rejected_macro_count": getattr(state, "rejected_macro_count", None),
        "tokens_used": getattr(state, "tokens_used", None),
        "sequence_counts": getattr(state, "sequence_counts", {}),
        "macro_usage_counts": getattr(state, "macro_usage_counts", {}),
        "macro_definitions": getattr(state, "macro_definitions", {}),
    }
    logger.info("Environment state snapshot:\n%s", safe_dump(snapshot))


def make_call(tool_name: str) -> ToolCall:
    return ToolCall(tool_name=tool_name)


def make_macro(name: str, description: str, steps: List[ToolCall]) -> Tool:
    return Tool(
        name=name,
        description=description,
        is_macro=True,
        token_cost=1,
        steps=steps,
    )


def build_deploy_health_steps() -> List[ToolCall]:
    return [
        make_call("deploy"),
        make_call("healthcheck"),
    ]


def build_full_action_for_task(task_id: str) -> ToolforgeAction:
    if task_id == "easy-deploy-notify":
        plan = [
            make_call("deploy"),
            make_call("healthcheck"),
            make_call("notify"),
        ]
    elif task_id == "easy-deploy-restart":
        plan = [
            make_call("deploy"),
            make_call("healthcheck"),
            make_call("restart"),
            make_call("healthcheck"),
            make_call("notify"),
        ]
    elif task_id == "easy-deploy-scale":
        plan = [
            make_call("deploy"),
            make_call("scale"),
            make_call("healthcheck"),
            make_call("notify"),
        ]
    else:
        plan = [make_call("healthcheck")]

    return ToolforgeAction(
        action_type="propose_plan",
        plan=plan,
        macro_proposal=None,
    )


def build_zero_fill_action() -> ToolforgeAction:
    return ToolforgeAction(
        action_type="propose_plan",
        plan=[make_call("scale")],
        macro_proposal=None,
    )


def build_partial_fill_action() -> ToolforgeAction:
    return ToolforgeAction(
        action_type="propose_plan",
        plan=[make_call("deploy")],
        macro_proposal=None,
    )


def build_wrong_tool_action() -> ToolforgeAction:
    return ToolforgeAction(
        action_type="propose_plan",
        plan=[make_call("wrong_tool")],
        macro_proposal=None,
    )


def build_premature_macro_creation_action() -> ToolforgeAction:
    macro_steps = build_deploy_health_steps()
    return ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=macro_steps,
        macro_proposal=make_macro(
            name="deploy_and_verify_macro",
            description="Deploy service and run a healthcheck.",
            steps=macro_steps,
        ),
    )


def build_repeat_sequence_seed(step_label: str) -> ToolforgeAction:
    macro_steps = build_deploy_health_steps()
    return ToolforgeAction(
        action_type="propose_plan",
        plan=macro_steps,
        macro_proposal=None,
    )


def build_mature_macro_creation_action() -> ToolforgeAction:
    macro_steps = build_deploy_health_steps()
    return ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=macro_steps,
        macro_proposal=make_macro(
            name="deploy_and_verify_macro",
            description="Deploy service and run a healthcheck.",
            steps=macro_steps,
        ),
    )


def build_wrong_macro_creation_action() -> ToolforgeAction:
    valid_plan = build_deploy_health_steps()
    return ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=valid_plan,
        macro_proposal=make_macro(
            name="broken_macro",
            description="Contains an invalid tool and should be rejected.",
            steps=[
                valid_plan[0],
                make_call("wrong_tool"),
            ],
        ),
    )


def build_macro_only_action() -> ToolforgeAction:
    return ToolforgeAction(
        action_type="propose_plan",
        plan=[make_call("deploy_and_verify_macro")],
        macro_proposal=None,
    )


def build_wrong_macro_name_action() -> ToolforgeAction:
    return ToolforgeAction(
        action_type="propose_plan",
        plan=[make_call("missing_macro_name")],
        macro_proposal=None,
    )


def build_macro_plus_atomic_action(task_id: str) -> ToolforgeAction:
    if task_id == "easy-deploy-notify":
        plan = [
            make_call("deploy_and_verify_macro"),
            make_call("notify"),
        ]
    elif task_id == "easy-deploy-restart":
        plan = [
            make_call("deploy_and_verify_macro"),
            make_call("restart"),
            make_call("healthcheck"),
            make_call("notify"),
        ]
    else:
        plan = [
            make_call("deploy_and_verify_macro"),
            make_call("scale"),
            make_call("healthcheck"),
            make_call("notify"),
        ]

    return ToolforgeAction(
        action_type="propose_plan",
        plan=plan,
        macro_proposal=None,
    )


def run_episode(
    title: str,
    scenario_builders: List[Tuple[str, str, Callable[[str], ToolforgeAction]]],
    summary_rows: List[Dict[str, Any]],
) -> None:
    banner(title)
    task_selector = TaskSelector(mode="eval")
    env = ToolforgeEnvironment(task_selector=task_selector, input_provider_factory=create_input_provider)
    obs = env.reset(episode_id=title.replace(" ", "-").lower(), mode="eval", difficulty="easy")

    logger.info("Episode reset complete")
    log_observation(obs)
    log_state(env)

    for step_index, (scenario_name, expectation, builder) in enumerate(scenario_builders, start=1):
        if obs.done:
            logger.warning("Episode ended before scenario '%s' could run.", scenario_name)
            summary_rows.append({
                "episode": title,
                "step": step_index,
                "scenario": scenario_name,
                "task_id": None,
                "reward": None,
                "done": True,
                "summary": "SKIPPED_EPISODE_DONE",
            })
            continue

        current_task = obs.current_task
        banner(f"{title} | Step {step_index} | Scenario: {scenario_name}")
        logger.info("Expected behavior: %s", expectation)
        logger.info("Current task id=%s", current_task.id)
        logger.info("Current task prompt=%s", current_task.prompt)
        logger.info("Current task required_slots=%s", current_task.required_slots)
        logger.info("Current task baseline_call_count=%s", current_task.baseline_call_count)

        action = builder(current_task.id)
        logger.info("Action payload:\n%s", safe_dump(action))

        obs = env.step(action)

        logger.info("Reward after step: %.4f", obs.reward if obs.reward is not None else 0.0)
        logger.info("Done after step: %s", obs.done)
        log_observation(obs)
        log_state(env)

        summary_rows.append({
            "episode": title,
            "step": step_index,
            "scenario": scenario_name,
            "task_id": current_task.id,
            "reward": obs.reward,
            "done": obs.done,
            "summary": (obs.metadata or {}).get("summary") if hasattr(obs, "metadata") else None,
            "plan_accepted": (obs.metadata or {}).get("plan_accepted") if hasattr(obs, "metadata") else None,
            "macro_decision": (obs.metadata or {}).get("macro_decision") if hasattr(obs, "metadata") else None,
        })


def main() -> None:
    check_llm_status()
    log_path = configure_logging()
    banner("REWARD TEST HARNESS START")
    logger.info("This harness focuses on reward-relevant scenarios and logs all step/observation details.")
    logger.info("Log file path: %s", log_path)

    summary_rows: List[Dict[str, Any]] = []

    run_episode(
        title="Episode A - Macro Maturity Threshold",
        scenario_builders=[
            ("repeat-sequence-seed-1", "First appearance of deploy->healthcheck sequence.", lambda _task_id: build_repeat_sequence_seed("seed-1")),
            ("repeat-sequence-seed-2", "Second appearance of deploy->healthcheck sequence.", lambda _task_id: build_repeat_sequence_seed("seed-2")),
            ("mature-macro-creation", "Macro proposal should now have prior sequence history available.", lambda _task_id: build_mature_macro_creation_action()),
        ],
        summary_rows=summary_rows,
    )

    run_episode(
        title="Episode B - Premature Macro and Macro Usage",
        scenario_builders=[
            ("premature-macro-creation", "Macro proposal occurs before enough prior sequence repetition; no recognition reward expected.", lambda _task_id: build_premature_macro_creation_action()),
            ("plan-with-macro-tool-call", "Plan uses the approved macro directly on the next task.", lambda _task_id: build_macro_only_action()),
            ("plan-with-macro-and-atomic", "Plan mixes an approved macro call with atomic tool calls.", build_macro_plus_atomic_action),
        ],
        summary_rows=summary_rows,
    )

    run_episode(
        title="Episode C - Slot Fill and Validation Paths",
        scenario_builders=[
            ("wrong-tool-call", "Stage 1 validation should reject the plan because the tool name does not exist.", lambda _task_id: build_wrong_tool_action()),
            ("zero-filled-slots", "Plan should produce zero useful slot fill with semantically unrelated valid tools.", lambda _task_id: build_zero_fill_action()),
            ("partially-filled-slots", "Plan should fill only part of the required workflow.", lambda _task_id: build_partial_fill_action()),
        ],
        summary_rows=summary_rows,
    )

    run_episode(
        title="Episode D - Full Fill and Macro Failure Paths",
        scenario_builders=[
            ("all-filled-slots", "Plan should fully satisfy the current task using atomic tools.", build_full_action_for_task),
            ("wrong-macro-creation-attempt", "Macro proposal should be rejected because it contains an invalid tool in its steps.", lambda _task_id: build_wrong_macro_creation_action()),
            ("wrong-macro-tool-call", "Validation should reject the plan because the macro name does not exist.", lambda _task_id: build_wrong_macro_name_action()),
        ],
        summary_rows=summary_rows,
    )

    banner("REWARD TEST SUMMARY")
    logger.info("Summary rows:\n%s", safe_dump(summary_rows))
    logger.info("Reward test harness complete. Detailed logs written to %s", log_path)


if __name__ == "__main__":
    main()
