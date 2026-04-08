"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional, Dict, Any

from openai import OpenAI

from toolforge_env import ToolforgeEnv, graders
from models import ToolCall, Tool, ToolforgeAction
IMAGE_NAME = os.getenv("IMAGE_NAME", "openenv-toolforge") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "toolforge_env")
MAX_STEPS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.6  # normalized score in [0, 1]

TASKS = [
    "easy",
    "medium",
    "hard",
]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent acting in the Toolforge environment.
    Return ONLY valid JSON matching the ToolforgeAction schema.

    Objective:
    - Maximize reward by completing task-required behavior with the relevant useful tool calls.

    Rules:
    - Use only tool names that appear in Available tools.
    - Keep the plan minimal and avoid unnecessary calls.
    - Use action_type="propose_plan" by default.
    - If proposing a macro, use action_type="propose_plan_with_macro" and include macro_proposal.
    - macro_proposal.steps must be an ordered sequence of at least 2 existing non-macro tools.
    - Do not use a newly proposed macro in the same step's plan.
    - If reusing an existing macro, do NOT send macro_proposal.
    - Never propose a macro name that already exists in Available tools.

    Macro policy:
    - Detect repetition by operation signature, not by exact wording.
    - Treat different service names/channels/contexts as the same pattern if tool order is the same.
    - Build a canonical pattern signature from tool order (for example: restart->healthcheck->notify).
    - Reuse an existing macro immediately when it matches the needed sequence.
    - Create a macro only when a contiguous ordered sequence has already repeated in prior steps.
    - Propose each macro only once. After approval, switch to action_type="propose_plan" with macro_proposal=null.
    - Prefer reusable patterns seen across task names and phases, not one-off service-specific steps.
    - Good reusable patterns include deploy->healthcheck->notify, restart->healthcheck->notify, rollback->healthcheck->notify, rollback->restart->healthcheck, scale->healthcheck->notify, and restart->deploy->healthcheck.
    - Some tasks are intentionally varied in wording; still group them by the same underlying tool-order signature.
    - This evaluator often treats each plan entry as filling at most one required slot.
    - Therefore, avoid macro-only one-entry plans for multi-slot tasks.
    - If task.required_slots has length N, prefer a plan with at least N entries, mixing macro calls with needed atomic calls.

    Naming:
    - Use short snake_case names that describe the operation pattern.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_task_list() -> List[str]:
    raw_tasks = os.getenv("MY_ENV_V4_TASKS", "")
    if raw_tasks.strip():
        return [task.strip() for task in raw_tasks.split(",") if task.strip()]

    single_task = os.getenv("MY_ENV_V4_TASK", "")
    if single_task.strip():
        return [single_task.strip()]

    return TASKS


def build_user_prompt(step: int, current_task: Any, available_tools: List[Dict[str, Any]], last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(
        f"""
        Step: {step}
        Current task: {current_task!r}
        Available tools: {json.dumps(available_tools)}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}

        Decision policy for macros:
        - First normalize the task into an operation signature using only tool order, ignoring service names and channel names.
        - If an existing macro matches the needed ordered sequence, reuse it now.
        - When reusing an existing macro, set action_type to propose_plan and set macro_proposal to null.
        - If the same normalized contiguous signature has appeared in earlier steps at least twice, and no existing macro covers it, propose a macro with propose_plan_with_macro.
        - Propose a given macro name only once; never re-propose an existing macro.
        - Never include a newly proposed macro in the same step plan.
        - If required_slots has length N, try to output at least N plan entries.
        - Avoid macro-only plans that underfill task requirements; include any additional atomic calls needed.
        - Prefer generic repeatable patterns across easy, medium, and hard tasks.
        - Keep the plan short while satisfying required task intent.

        Return a ToolforgeAction whose `plan` uses only currently available tools.
        """
    ).strip()


def build_fallback_action(available_tools: List[Dict[str, Any]], current_task: str) -> ToolforgeAction:
    fallback_tool = available_tools[0]["name"] if available_tools else "noop"
    return ToolforgeAction(
        action_type="propose_plan",
        plan=[
            ToolCall(tool_name=fallback_tool)
        ],
        macro_proposal=None
    )


def get_model_action(
    client: OpenAI,
    step: int,
    current_task: str,
    available_tools: List[Dict[str, Any]],
    last_reward: float,
    history: List[str],
) -> ToolforgeAction:
    user_prompt = build_user_prompt(step, current_task, available_tools, last_reward, history)
    try:    
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nRespond ONLY with valid JSON matching the ToolforgeAction schema. No markdown, no explanation."},
                {"role": "user", "content": user_prompt + 
                 f"\n\nSchema:\n{json.dumps(ToolforgeAction.model_json_schema(), indent=2)}" + 
                 f"\n\nToolSchema:\n{json.dumps(Tool.model_json_schema(), indent=2)}" +
                 f"\n\nToolCallSchema:\n{json.dumps(ToolCall.model_json_schema(), indent=2)}"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = completion.choices[0].message.content
        if raw is None:
            return build_fallback_action(available_tools, current_task)
        raw = raw.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return ToolforgeAction(**json.loads(raw))

    except Exception as e:
        return build_fallback_action(available_tools, current_task)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await ToolforgeEnv.from_docker_image(IMAGE_NAME, env_vars={"HF_TOKEN": API_KEY})
    grader = graders.EpisodeGrader()
    task_list = get_task_list()

    try:
        for task_name in task_list:
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = await env.reset(task_id=task_name) # OpenENV.reset()
                obs = result.observation
                task = obs.current_task
                available_tools = obs.available_tools
                history = [
                    f"EpisodeStart|task_id {getattr(task, 'id', 'unknown')}|difficulty {getattr(task, 'difficulty', 'unknown')}|prompt {getattr(task, 'prompt', '')}"
                ]
                last_reward = 0.0

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    action = get_model_action(client, step, task.prompt, available_tools, last_reward, history)
                    result = await env.step(action)
                    obs = result.observation
                    task = obs.current_task
                    available_tools = obs.available_tools

                    reward = result.reward or 0.0
                    done = result.done
                    error = obs.metadata.get("summary") if isinstance(obs.metadata, dict) else None

                    rewards.append(reward)
                    steps_taken = step
                    last_reward = reward

                    log_step(
                        step=step,
                        action=action.model_dump_json(),
                        reward=reward,
                        done=done,
                        error=error,
                    )

                    history.append(f"{action.model_dump_json()}|Step {step}|reward {reward:+.2f}|task_id {getattr(task, 'id', 'unknown')}|difficulty {getattr(task, 'difficulty', 'unknown')}")

                    if done:
                        break

                score = grader.grade(obs)
                success = score >= SUCCESS_SCORE_THRESHOLD
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
