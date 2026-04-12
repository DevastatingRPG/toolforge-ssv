# CLAUDE.md — ToolForge Environment

## Project Context

ToolForge is an OpenEnv-native benchmark environment for training LLM agents to recognize repetitive DevOps tool-call patterns and abstract them into reusable **macro tools**. The dual objective is correctness (did the agent solve the task?) and efficiency (did it compress repeated patterns to save tokens?).

This is a hackathon submission targeting the Meta & HuggingFace AI Hackathon.

---

## Working Directory

The primary workspace is `openenv-SSV/`. All edits happen here.

The `openenv-course/OpenEnv` folder (sibling directory) is a reference-only resource for OpenEnv API standards and helper methods. Do not modify it.

---

## Agent Instructions (from `.agents/workflows/instructions.md`)

1. **Missing Info Tracking:** If details are missing while coding, write a comment at the very top of the edited file. Remove it once the user clarifies.

2. **Never touch untaught details** without asking the user first.

3. **Self-Updating Instructions:** If the user gives a new general rule, add it to `.agents/workflows/instructions.md` immediately.

4. **Coding Conventions:**
   - Clean, modular, well-documented code (PEP 8)
   - Type hints on all function signatures
   - `import logging` — every module must use proper logging
   - Heavy commenting required:
     - **Classes**: docstring describing purpose
     - **Methods**: "butterfly comments" (prominent block comments) outlining purpose
     - **Variables**: inline comment explaining what each variable is used for
   - Descriptive, meaningful names for everything

---

## Repository Layout

```
openenv-SSV/
├── openenv.yaml                          # Manifest: task IDs, grader references
├── inference.py                          # Benchmark runner (must use OpenAI client)
├── models.py                             # Shared Pydantic models (source of truth)
├── graders.py                            # Package-level grader re-export
├── README.md
├── submission.md
└── server/
    ├── app.py                            # FastAPI entrypoint via openenv
    ├── toolforge_env_environment.py      # Core env: reset(), step(), state
    ├── tools.py                          # Atomic tool definitions (10 tools)
    ├── slots.py                          # 23 DevOps semantic slot definitions
    ├── graders.py                        # grade_easy / grade_medium / grade_hard
    └── evaluation/
    │   ├── pipeline.py                   # Orchestrates 4-stage evaluation
    │   ├── plan_evaluator.py             # Stage logic + reward math
    │   ├── llm_eval_prompts.py           # LLM judge prompts
    │   └── tool_slot_mappings.py         # Tool → slot mappings
    └── inputs/
        ├── base.py                       # InputProvider interface
        ├── factory.py                    # Input provider factory
        └── simulated/
            ├── tasks.py                  # Task banks: easy / medium / hard
            ├── task_selector.py          # Routes "easy"/"medium"/"hard" → task groups
            └── data_loader.py            # Data loading utilities
```

---

## Core Data Models (`models.py`)

| Model | Purpose |
|---|---|
| `ToolCall` | Single tool invocation (`tool_name`) |
| `Tool` | Atomic or macro tool (`is_macro`, optional `steps`) |
| `ToolForgeAction` | Agent action: `action_type`, `plan`, optional `macro_proposal` |
| `Task` | Task with `prompt`, `difficulty`, `required_slots`, `baseline_call_count` |
| `ToolForgeObservation` | Env response: `current_task`, `available_tools`, `reward`, `done` |
| `ToolForgeState` | Internal episode state: task queue, macros, history, grading accumulator |
| `EpisodeGradingState` | Accumulated episode metrics for grader scoring |

---

## Environment Logic (`toolforge_env_environment.py`)

- Extends `openenv.core.env_server.interfaces.Environment`
- `MAX_EPISODE_STEPS = 100` hard limit

### `reset(task_id, episode_id)`
TaskSelector routes `easy/medium/hard` → internal task group → loads task list → initializes state → returns first observation.

### `step(action: ToolForgeAction)`
1. Validates action structure
2. Analyzes plan (unknown tools, call count)
3. Runs 4-stage evaluation pipeline
4. Updates sequence counts (for macro recognition)
5. Processes macro proposal lifecycle
6. Advances to next task or ends episode
7. Returns observation with scalar reward

---

## Evaluation Pipeline (4 Stages)

### Stage 1: Sanity Validation
- Rejects empty plans or unknown tool names
- Failure → penalty `-0.2`, short-circuit

### Stage 2: Semantic Slot Judgment (LLM)
- Calls `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router
- Fallback: rule-based parser if LLM fails
- Classifies each tool call: relevant / unnecessary / harmful
- Harmful → short-circuit; outputs `slot_ratio`, `slot_score`

### Stage 3: Macro Bonuses
- **Macro creation** (max `+0.20`): Sequence seen ≥2 times prior + plan semantically adequate
- **Macro usage** (max `+0.05`): Used an accepted macro when `slot_ratio ≥ 0.65`
- **Macro miss penalty** (`-0.02` to `-0.10`): Wrote out atomic steps when a macro was available

### Stage 4: Efficiency Score
- Activates only when `slot_ratio == 1.0`
- Compares actual plan length vs `baseline_call_count`
- Fewer calls → higher score; bounded `[0.0, 0.5]`

### Final Reward Composition
| Component | Range |
|---|---|
| Slot score (slot_ratio < 0.65) | [-0.15, 0.0] |
| Slot score (slot_ratio ≥ 0.65) | [0.0, 0.25] |
| Macro creation bonus | [0.0, 0.20] |
| Macro usage bonus | [0.0, 0.05] |
| Macro miss penalty | [-0.10, 0.0] |
| Efficiency score | [0.0, 0.50] |
| **Final (clamped)** | **[-0.2, 1.0]** |

---

## Grader Logic (`server/graders.py`)

Graders compute a normalized `[0.01, 0.99]` episode score (separate from step reward):

| Signal | Weight |
|---|---|
| accuracy (correct_plan_count / steps) | 40% |
| token_optimization (avg efficiency on correct steps) | 30% |
| macro_creation quality | 20% |
| macro_usage quality | 10% |

Referenced by `openenv.yaml` as `server.graders.grade_easy/medium/hard`.

---

## Atomic Tools (10 total)
`deploy`, `patch`, `healthcheck`, `run_tests`, `ping`, `notify`, `pagerduty_alert`, `rollback`, `scale`, `restart`

## Semantic Slots (23 total)
Cover deployment, patch, rollback, scaling, and restart phases across execution / verification / notification dimensions.

---

## Task Difficulty

| Level | Pattern | Internal Group |
|---|---|---|
| Easy | Repetitive 3-slot patterns (deploy→healthcheck→notify) | easy-deployment-sprints, easy-resource-management, easy-rollback-drills |
| Medium | Multi-slot reasoning, lower pattern overlap | medium-traffic-readiness, medium-incident-response |
| Hard | Multi-phase complex infrastructure scenarios | hard-project-legacy-migration |

---

## Macro Lifecycle

1. Agent submits `action_type="propose_plan_with_macro"` with a `macro_proposal`
2. Validation: min 2 steps, all atomic (no nested macros), no duplicate tool names, must pass sanity validation
3. If valid → added to `available_tools` + sequence tracking
4. Subsequent steps: macro creation bonus requires the sequence to have been seen ≥2 times previously
5. Macros are surfaced prominently in observations for the agent to reuse

---

## Inference Script Requirements (`inference.py`)

- Must use `openai` Python client
- Credentials from: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Emits structured stdout: `[START]`, `[STEP]`, `[END]` lines
- Final scores clamped to `(0.01, 0.99)`
- Must complete < 20 minutes

---

## Hackathon Constraints

- Must pass `openenv validate`
- Docker image must build and run cleanly
- All grader outputs clamped to `(0.01, 0.99)` — never exact `0.0` or `1.0`
- Step reward bounded `[-0.2, 1.0]`
- Endpoints: `POST /reset`, `POST /step`, `GET /state`, `GET /schema`
- Max concurrent envs: 1
- Resource limit: 2 vCPU, 8GB RAM

---

## Key Thresholds & Constants

| Constant | Value |
|---|---|
| `slot_threshold` (macro gate) | 0.65 |
| `MAX_EPISODE_STEPS` | 100 |
| Macro min steps | 2 |
| Sequence repeat for creation bonus | ≥ 2 prior occurrences |
| LLM model (slot judge) | Qwen/Qwen2.5-72B-Instruct |
