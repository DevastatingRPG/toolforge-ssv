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
import pprint
import textwrap
from typing import List, Optional, Dict, Any

from openai import OpenAI

from toolforge_env import ToolforgeAction, ToolforgeEnv
from toolforge_env.models import ToolCall
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 400
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent acting in the Toolforge environment.
    Produce a valid ToolforgeAction object that matches the provided schema.
    Build a concise plan using only the tools listed in the prompt.
    Prefer `action_type="propose_plan"` unless a macro proposal is genuinely helpful.
    Keep reasoning brief and make every plan step actionable.
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


def build_user_prompt(step: int, current_task: str, available_tools: List[str], last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current task: {current_task!r}
        Available tools: {json.dumps(available_tools)}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Return a ToolforgeAction whose `plan` uses only the available tools.
        """
    ).strip()


def build_fallback_action(available_tools: List[str], current_task: str) -> ToolforgeAction:
    fallback_tool = available_tools[0] if available_tools else "noop"
    return ToolforgeAction(
        action_type="propose_plan",
        plan=[
            ToolCall(tool_name=fallback_tool, params={"task": current_task}, token_cost=0)
        ],
        reasoning="Fallback action because structured model parsing failed.",
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
        # completion = client.beta.chat.completions.parse(
        #     model=MODEL_NAME,
        #     messages=[
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     response_format=ToolforgeAction,
        #     temperature=TEMPERATURE,
        #     max_tokens=MAX_TOKENS,
        # )
        # parsed = completion.choices[0].message.parsed
        # if parsed is None:
        #     raise ValueError("Model returned no parsed ToolforgeAction.")
        # return parsed
    
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nRespond ONLY with valid JSON matching the ToolforgeAction schema. No markdown, no explanation."},
                {"role": "user", "content": user_prompt + f"\n\nSchema:\n{json.dumps(ToolforgeAction.model_json_schema(), indent=2)}"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = completion.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return ToolforgeAction(**json.loads(raw))

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return build_fallback_action(available_tools, current_task)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await ToolforgeEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=TASK_NAME, mode="eval", difficulty="easy") # OpenENV.reset()
        obs = result.observation
        task = obs.current_task
        available_tools = obs.available_tools
        # history = obs.history
        history = []
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, step, task.prompt, available_tools, last_reward, history)

            result = await env.step(action)
            obs = result.observation

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

            history.append(f"Step {step}: {action.model_dump_json()} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
