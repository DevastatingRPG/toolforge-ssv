# ToolForge Submission Report

## How This Document Is Organized
This document is structured as a technical submission overview for ToolForge. It starts with the project motivation and benchmark framing, then explains the runtime and deployment path, repository layout, core models, environment design, API surface, reward pipeline, grader design, and testing/validation process.

The intended audience is a hackathon reviewer, teammate, or maintainer who needs to understand how the environment works end to end without reverse-engineering the codebase.

# ToolForge: Agentic Efficiency through Macro Extraction
ToolForge is about saving tokens by abstracting tool sequences into reusable **macro tools**, significantly reducing both "thinking" (reasoning) tokens and "calling" (API overhead) tokens. 
### Core Vision and Impact
In the current era of agentic AI, models use tools more extensively than ever before. This creates a practical cost and latency problem: models repeatedly spend their reasoning budget to plan and execute the same multi-step tool patterns from scratch.
This Proof-of-Concept (POC) demonstrates that we can train models to decrease token usage by:
1.  **Recognizing** repeated, ordered tool-call patterns.
2.  **Compressing** those patterns into reusable macro abstractions.
3.  **Reusing** those abstractions in subsequent steps to solve complex workflows.

The goal is not just faster task completion. The deeper objective is to teach models to **spend less reasoning budget on workflows they have already effectively learned** and reserve more of their token budget for genuinely new planning problems.

### Domain Versatility
While this specific POC is built around a **DevOps workflow simulation**, the underlying abstraction is not limited to that domain. We believe the same principles can be expanded to any tool-using environment where agentic models repeatedly solve structured workflows—from data engineering and cloud management to automated software testing and administrative automation.



## The Benchmark Objectives
ToolForge evaluates agentic performance across three primary pillars:
*   **Correctness**: Ensuring high-fidelity task completion through semantic validation.
*   **Abstraction**: Measuring the efficiency of `macro_creation` in identifying valid patterns.
*   **Reuse**: Tracking `macro_usage` to verify that models effectively reduce their total call count and reasoning overhead.
## Technical Architecture (OpenEnv)
ToolForge follows the standard OpenEnv interaction contract to ensure seamless evaluation and portability:
*   **`reset()`**: Initializes a fresh episode and provides the initial task and toolset.
*   **`step()`**: Processes agent plans, executes evaluations, and returns rewards/observations.
*   **`openenv.yaml`**: The manifest-driven source of truth for task exposure and grader mapping.
*   **`inference.py`**: The standardized submission-time script that drives the agent loop.
## Evaluation Difficulty
ToolForge exposes three specific benchmark task IDs to test varying levels of pattern density:
1.  **`easy`**: High-repetition environments designed for quick macro identification and practice.
2.  **`medium`**: Mixed operational goals with lower pattern overlap to test reasoning versus abstraction.
3.  **`hard`**: Complex, multi-phase projects where context must be maintained across long-horizon scenarios.

## Running, Deployment, and Modes
### Local Run
ToolForge can be run locally as an OpenEnv environment server. The key files for local execution are:
- `openenv.yaml` for environment metadata and benchmark task definitions,
- `server/app.py` for the FastAPI server entrypoint,
- `inference.py` for submission-style execution across benchmark tasks.

Typical local workflow:
1. Start the environment server locally or through Docker.
2. Run `inference.py` against the benchmark task ids.
3. Inspect reward logs, evaluation outputs, and grader behavior.

### Docker and OpenEnv Usage
The project is packaged as a Docker-backed OpenEnv environment. The runtime manifest is declared in `openenv.yaml`, including:
- environment name,
- runtime type,
- server app path,
- benchmark task entries,
- grader references.

The environment is exposed through `server.app:app`, and the OpenEnv client interacts with it through `reset()` and `step()`.

### Hugging Face Space Deployment
ToolForge is intended to run as a Hugging Face Space-backed OpenEnv environment. In that deployment model:
- the Space hosts the environment server,
- benchmark tasks are declared in `openenv.yaml`,
- the inference runner executes the benchmark tasks `easy`, `medium`, and `hard`.

### Required Environment Variables / Secrets
The LLM-backed parts of the system depend on environment variables, especially when running in a hosted Space:
- `HF_TOKEN` or `API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- optional image/runtime configuration values used by `inference.py`

The important operational point is that the server-side evaluator may also require access to the LLM token if live semantic slot judgment is enabled. In hosted deployments, these values should be configured through Space secrets rather than relying only on a local `.env` file.

### Eval Mode vs Train Mode
ToolForge conceptually supports two different usage patterns.

#### Eval Mode
In eval mode, the environment uses fixed benchmark task ids:
- `easy`
- `medium`
- `hard`

These resolve deterministically to internal task groups. This is the mode used by `inference.py` for benchmark submission and structured evaluation.

The objective in eval mode is consistency:
- deterministic task-group selection,
- reproducible benchmark execution,
- fixed grader routing.

#### Train Mode
In train mode, the intended usage pattern is richer and more episode-oriented.

Here the environment can be used to run repeated episodes in which the agent:
- encounters repeated workflows,
- maintains and accumulates accepted macros,
- is evaluated step-by-step through the reward pipeline,
- gradually learns to abstract recurring patterns.

The purpose of train mode is not just final scoring, but repeated interaction and macro learning over time.

## Repository and Folder Structure
Below is the practical structure of the project and what the major files do.

```text
toolforge_env/
+-- openenv.yaml
+-- inference.py
+-- graders.py
+-- models.py
+-- README.md
+-- submission.md
+-- test_reward.py
+-- server/
�   +-- app.py
�   +-- graders.py
�   +-- tools.py
�   +-- slots.py
�   +-- toolforge_env_environment.py
�   +-- evaluation/
�   �   +-- __init__.py
�   �   +-- pipeline.py
�   �   +-- plan_evaluator.py
�   �   +-- llm_eval_prompts.py
�   +-- inputs/
�       +-- simulated/
�           +-- tasks.py
�           +-- task_selector.py
```

### Important Modules
- `models.py`
  - shared typed models used by the environment, evaluator, grader, and inference logic
- `inference.py`
  - submission-time benchmark runner using the OpenAI client and OpenEnv environment client
- `graders.py`
  - package-level re-export layer for grader compatibility
- `server/graders.py`
  - benchmark grader callables referenced by `openenv.yaml`
- `server/app.py`
  - FastAPI/OpenEnv server entrypoint
- `server/toolforge_env_environment.py`
  - core environment logic including reset, step, task progression, macro persistence, and state updates
- `server/evaluation/`
  - home of the structured evaluation pipeline and evaluator logic
- `server/tools.py`
  - tool inventory available to the agent
- `server/slots.py`
  - semantic slot definitions used during judging
- `server/inputs/simulated/tasks.py`
  - benchmark task banks for easy, medium, and hard internal groups
- `server/inputs/simulated/task_selector.py`
  - deterministic mapping from benchmark task ids to internal task lists
- `test_reward.py`
  - local reward harness for inspecting step-level evaluation behavior

## Core Data Models
The project uses `models.py` as the shared contract layer.

### Action and Plan Layer
#### `ToolCall`
Represents one tool invocation in a proposed plan. In the current design, the minimal identifier is the tool name.

#### `Tool`
Represents either:
- an atomic tool, or
- a macro tool composed of multiple `ToolCall` steps.

This model is also used when a macro is proposed or when a macro becomes part of the available tool inventory.

#### `ToolforgeAction`
Represents the agent action sent to the environment. It includes:
- `action_type`
- `plan`
- optional `macro_proposal`

This is the main action payload used by `step()`.

### Task and Observation Layer
#### `Task`
Represents one operational goal. It contains:
- task id,
- prompt,
- difficulty,
- `required_slots`,
- baseline call-count metadata.

#### `ToolforgeObservation`
Represents the environment response after `reset()` or `step()`. It exposes:
- current task,
- available tools visible to the agent,
- reward,
- done flag,
- optional metadata.

#### `ToolForgeState`
Tracks environment state across the episode. It owns:
- current task and task queue,
- completed tasks,
- tool inventory,
- accepted macros,
- call history,
- macro definitions and usage counts,
- sequence history used for macro recognition,
- episode-level counters.

### Evaluation Layer
The evaluation pipeline uses typed intermediate results to keep stage outputs explicit.

These include:
- validation result
- slot judgment result
- token / efficiency result
- pipeline result

Together they allow the environment to distinguish:
- structural failure,
- semantic failure,
- incomplete but threshold-clearing plans,
- fully correct plans,
- macro and efficiency contributions.

### Grading Layer
The codebase also includes episode-level grading support. Where present, an episode grading state accumulator is used to track run-level benchmark signals such as:
- accuracy,
- efficiency,
- macro creation,
- macro usage,
- rejections or harmful behavior.

This is distinct from the step reward signal.

## Environment Design
`ToolforgeEnvironment` is the core execution engine. It owns task progression, tool exposure, macro lifecycle, and the link between an agent action and the evaluation pipeline.

### `reset()`
At reset time, the environment:
- creates a fresh episode state,
- resolves the incoming benchmark task id,
- maps `easy`, `medium`, or `hard` to an internal task group,
- loads the corresponding task list through the simulated input provider,
- initializes the first current task,
- rebuilds the available atomic tool set,
- preserves accepted macros if persistence is enabled by the current design,
- resets sequence and usage tracking for the new episode,
- returns the first observation.

### `step()`
At each step, the environment:
- validates the incoming action type,
- records the submitted tool calls into episode history,
- builds the available-tool registry,
- calls the evaluation pipeline,
- updates sequence counts after evaluation,
- updates macro usage counters,
- processes macro approval or rejection,
- advances to the next task or ends the episode,
- returns the next observation with the scalar reward.

### Task Selection
The public benchmark ids are:
- `easy`
- `medium`
- `hard`

These do not directly correspond to inner prompt ids such as `e-dep-1` or `m-inc-3`. Instead, the selector maps them to internal task groups such as:
- `easy-deployment-sprints`
- `medium-traffic-readiness`
- `hard-project-legacy-migration`

This allows the submission-facing interface to stay simple while the internal task banks remain descriptive.

### Macro Management
The environment owns macro lifecycle management:
- macro proposals are validated and either approved or rejected,
- accepted macros are stored in state,
- macro definitions are persisted for later recognition and usage checks,
- macros are exposed through `available_tools`,
- macro ordering is handled so macros can be surfaced prominently to the agent.

### Available Tools Exposure
The environment converts `Tool` models into prompt-friendly dictionaries before returning them in the observation. This is what the inference script passes back to the LLM in subsequent steps.

## API Surface
### Environment Interaction
The practical environment API is the standard OpenEnv interaction loop:
- `reset(task_id=...)`
- `step(action)`

### Action Payload
A valid action contains:
- `action_type`
- `plan`
- optional `macro_proposal`

### Observation Payload
An observation contains:
- `current_task`
- `available_tools`
- `reward`
- `done`
- optional metadata

### Evaluation Pipeline Entry Point
The evaluator is invoked through the server-side pipeline entrypoint in the evaluation package. It is responsible for turning a proposed plan into structured stage outputs and a final bounded step reward.

### Grader API
The manifest references grader callables via `openenv.yaml`, currently through entries such as:
- `server.graders.grade_easy`
- `server.graders.grade_medium`
- `server.graders.grade_hard`

These are benchmark-time scoring functions, distinct from the environment step reward.

### Inference Logging Contract
`inference.py` emits structured stdout lines in the required format:
- `[START]`
- `[STEP]`
- `[END]`

These lines encode:
- task id,
- action,
- reward,
- done flag,
- error state,
- final score.

## Reward Philosophy
ToolForge�s reward design is built around a simple principle:
- correctness first,
- abstraction second,
- efficiency third,
- harmful or invalid behavior should not be rewarded.

The goal is not merely to reward tool use, but to reward **correct tool use that becomes more abstract and efficient over time**.

### Stage 1: Validation
The first stage checks whether the plan is structurally valid.

This includes checks such as:
- empty plan,
- invalid tool names,
- malformed plan structure.

If this stage fails, the environment applies an immediate failure path and short-circuits later stages.

### Stage 2: Semantic Slot Judgment
The second stage uses semantic slot evaluation to determine whether the proposed plan actually addresses the task.

This stage evaluates:
- which required semantic slots were filled,
- the resulting `slot_ratio`,
- whether harmful or harmless calls were made.

#### Harmful Calls
A harmful call is a tool call that is semantically dangerous in the context of the task. It is not merely irrelevant; it represents an action that can actively move the system in the wrong direction.

Examples include:
- rollback-like behavior when the task is clearly asking for forward deployment,
- a call that undoes required progress,
- a semantically damaging step that should immediately invalidate the attempt.

If harmful calls are detected, the pipeline short-circuits immediately.

#### Harmless Calls
A harmless call is a tool call that may be unnecessary or redundant but is not actively damaging.

Examples include:
- an extra but non-destructive verification step,
- a redundant notification,
- a superfluous but safe operational step.

Harmless calls may reduce reward quality indirectly, but they do not trigger immediate catastrophic failure in the way harmful calls do.

#### `slot_ratio`
`slot_ratio` is the ratio of semantic slots filled to semantic slots required for the current task.

Conceptually:
- `slot_ratio = filled_slots / required_slots`

This is the main signal used to determine how semantically complete the plan is.

#### `slot_score`
`slot_score` is the bounded semantic score derived from `slot_ratio`.

It translates slot completion quality into a reward-stage contribution that can be:
- negative for poor semantic performance,
- neutral around the threshold boundary,
- positive for strong semantic completion.

### Stage 3: Macro Rewards
This stage uses two explicit concepts:
- `macro_creation`
- `macro_usage`

#### `macro_creation`
This rewards the agent for introducing a good macro based on historically repeated ordered sequences. It is larger than `macro_usage` because the benchmark wants to teach abstraction discovery, not only reuse.

`macro_creation` depends on:
- prior repeated sequence history,
- semantic adequacy of the current plan,
- ordered contiguous sequence matching.

#### `macro_usage`
This rewards the agent for correctly using an approved macro.

It is intentionally smaller than `macro_creation` because the main operational benefit of macro use is already reflected through efficiency.

#### Threshold Gating and `slot_threshold`
Macro rewards are gated by `slot_threshold`.

`slot_threshold` is the minimum semantic quality required before the benchmark starts rewarding abstraction-related behavior. Its significance is important:
- below the threshold, the plan is too semantically weak to deserve macro credit,
- above the threshold, the plan is strong enough to reward useful abstraction signals,
- at full completion, both abstraction and efficiency can be rewarded together.

This prevents the agent from learning to optimize around macros before it has learned to solve the task itself.

The gating behavior is:
- below `slot_threshold`: no macro bonus,
- above threshold but below full completion: partial macro eligibility,
- at full completion: both `macro_creation` and `macro_usage` are allowed.

### Stage 4: Tool Efficiency
The efficiency stage uses **count-based efficiency**, not the older idea of per-tool token burn.

Important properties:
- baseline comes from `baseline_call_count` on the task,
- the baseline plan gets a mid-level positive efficiency score,
- fewer calls than baseline yield a higher score,
- more calls than baseline yield a lower score.

This keeps the benchmark aligned with plan compactness rather than arbitrary per-tool pricing.

## Reward Calculation Flow
The reward pipeline follows this high-level flow:
1. validation
2. semantic slot judgment
3. macro bonus computation
4. efficiency scoring
5. final bounded step reward

### Reward Range
The step reward is bounded between:
- `-0.2`
- `1.0`

### Short-Circuit Conditions
The pipeline short-circuits when:
- validation fails,
- harmful calls are detected,
- semantic quality is too weak for later-stage reward activation.

### Stage Activation Rules
- Validation failure ends scoring early.
- Harmful semantic judgment also ends scoring early.
- Macro bonuses activate only once slot quality clears the threshold region.
- Efficiency activates only when the plan is fully correct.

### Final Reward Composition
The final reward is bounded and additive across active stages.

This means:
- invalid or harmful plans cannot accidentally receive efficiency or macro credit,
- macro bonuses only appear when correctness is already meaningful,
- efficiency only matters once the task is semantically solved.

## Grader Design
The grader is separate from the step reward.

### Role of the Grader
The environment step reward is the training-time signal used during interaction.

The grader is the benchmark-time scoring mechanism used after a full episode/run is complete. It is responsible for assigning a final run-level score for submission evaluation.

### Manifest Integration
Graders are referenced in `openenv.yaml` through the benchmark task definitions. For example:
- `easy -> server.graders.grade_easy`
- `medium -> server.graders.grade_medium`
- `hard -> server.graders.grade_hard`

### Clamping Requirement
Per hackathon validation requirements, grader outputs must be clamped to:
- `(0.01, 0.99)`

The same principle is also applied in `inference.py` for final score reporting so that no code path emits exact `0.0` or `1.0`.

### What the Grader Looks At
The intended grader design is based on full-episode performance, including signals such as:
- accuracy across the run,
- token optimization,
- macro creation quality,
- macro usage quality,
- any episode-level grading accumulator stored in state.

This separation is deliberate:
- step reward teaches the agent,
- grader evaluates the completed run.

### How to Use the Grader
For benchmark submission, the grader is not invoked manually in isolation. Instead:
1. `openenv.yaml` declares which grader function is associated with each benchmark task id,
2. the run is executed through the benchmark runner,
3. the corresponding grader is called on the completed episode/run,
4. the output is clamped and used as the final benchmark score.

Locally, grader behavior can be tested by exercising completed episode outputs and checking that:
- the expected grader callable resolves,
- edge cases still clamp correctly,
- no path returns exact `0.0` or `1.0`.

## Testing and Validation
ToolForge includes local utilities for reward and task validation.

### Local Reward Testing
`test_reward.py` is used to inspect reward behavior across representative scenarios. This includes:
- malformed actions,
- partial slot filling,
- macro proposal paths,
- macro usage paths,
- end-to-end step reward logging.

### Task Validation
The repository also includes helpers such as task-generation and validation scripts to check that task banks remain structurally consistent with:
- tool availability,
- semantic slots,
- baseline call counts.

### Submission Validation Concerns
Before submission, the project was checked for:
- manifest structure,
- presence of benchmark tasks with graders,
- inference stdout contract,
- score clamping requirements,
- server import paths,
- benchmark task selection.

### Grader Testing Intention
The grader is treated as a separate benchmark component and should be tested independently from step reward logic. This includes:
- ensuring callable discovery from `openenv.yaml`,
- validating score clamping,
- ensuring edge cases never return exact `0.0` or `1.0`.

## Future Work
The current implementation is a strong proof of concept, but several improvements are intentionally left for future iterations.

Planned future work includes:
- parameter-aware tool-call validation and richer payload checking,
- stronger LLM judge models and more robust semantic evaluation backends,
- more tools for broader workflow coverage,
- more tasks across all difficulty levels,
- further benchmark expansion beyond the current simulated DevOps domain.

## Closing Note
ToolForge is designed as an OpenEnv-native benchmark for studying how agents move from atomic tool use to structured abstraction. The core benchmark question is not only whether an agent can complete tasks, but whether it can learn to **compress repeated workflows into reusable macro tools while preserving correctness**.
