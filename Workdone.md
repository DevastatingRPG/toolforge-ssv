# ToolForge Current Work Status

This document tracks the current implementation state of the ToolForge environment project.

## Overall Plan

The broader plan is to build an OpenEnv-compatible benchmark where:
- an agent receives a task prompt
- submits a structured plan made of tool calls
- is judged through a multi-stage evaluation pipeline
- eventually learns reusable abstractions such as macro tools
- is scored on both correctness and efficiency

The current implementation is focused on getting the skeleton and first judge wiring in place before deeper logic is added.

## What Has Been Implemented

### 1. Standalone OpenEnv project structure
The project has a standalone environment package with:
- package metadata
- OpenEnv manifest
- client
- server app
- environment class
- models
- task and tool definitions

### 2. Typed models
Implemented in:
- `toolforge_env/toolforge_env/models.py`

Current model coverage includes:
- tool calls
- tools
- tasks
- macro proposals
- action
- observation
- state

Judge-related result types have also been added:
- `ValidationResult`
- `ToolEvaluation`
- `SlotJudgmentResult`
- `RewardResult`
- `TokenCostResult`
- `PipelineResult`

### 3. Atomic tools
Implemented in:
- `toolforge_env/toolforge_env/server/tools.py`

Current atomic tools:
- deploy
- healthcheck
- notify
- rollback
- scale
- restart

Each tool has:
- description
- parameter schema
- token cost

### 4. Easy task bank
Implemented in:
- `toolforge_env/toolforge_env/server/tasks.py`

Current tasks:
- easy-deploy-notify
- easy-deploy-restart
- easy-scale-notify
- easy-restart-notify
- easy-rollback-notify
- easy-deploy-scale

Each task now includes:
- prompt
- difficulty
- required steps
- core steps
- required semantic slots
- baseline token cost

### 5. Slot library
Implemented in:
- `toolforge_env/toolforge_env/server/slots.py`

Current slot vocabulary:
- `DEPLOYMENT_ACTION`
- `VERIFICATION_ACTION`
- `NOTIFICATION_ACTION`
- `ROLLBACK_ACTION`
- `SCALING_ACTION`
- `CONFIGURATION_ACTION`

### 6. Simulated user stub
Implemented in:
- `toolforge_env/toolforge_env/server/user_sim.py`

Current state:
- class exists
- easy mode supported
- deterministic exact-match evaluation exists

Important:
- it is not yet meaningfully integrated into `step()`

### 7. Judge pipeline stages

#### Stage 1 — validator
Implemented in:
- `toolforge_env/toolforge_env/server/validator.py`

This is real logic, not placeholder.

Checks:
- empty plan
- invalid tool
- missing required param
- extra param

This stage short-circuits the pipeline when invalid.

#### Stage 2 — slot judge
Implemented in:
- `toolforge_env/toolforge_env/server/slot_judge.py`

This is placeholder logic.

Current behavior:
- returns success-shaped output
- marks all required slots filled
- `task_complete = True`

No real LLM/API call yet.

#### Stage 3 — reward calculator
Implemented in:
- `toolforge_env/toolforge_env/server/reward_calculator.py`

This is placeholder logic.

Current behavior:
- returns fixed perfect reward values

#### Stage 4 — token cost calculator
Implemented in:
- `toolforge_env/toolforge_env/server/token_calculator.py`

This is placeholder logic.

Current behavior:
- returns fixed perfect efficiency values

### 8. Judge orchestrator
Implemented in:
- `toolforge_env/toolforge_env/server/judge_pipeline.py`

This is the central judging entrypoint.

Current behavior:
- runs validator first
- short-circuits on validation failure
- otherwise runs slot judge
- then reward calculator
- then token calculator
- computes final score:
  - 70% correctness
  - 30% efficiency
- clamps final score to `[0.0, 1.0]`

### 9. Environment integration
Implemented in:
- `toolforge_env/toolforge_env/server/toolforge_environment.py`

Current `step()` behavior:
- increments `step_count`
- records tool calls
- accumulates `tokens_used`
- calls `run_judge_pipeline(...)`
- uses `pipeline_result.final_score` as observation reward
- stores lightweight metadata:
  - `summary`
  - `passed_validation`
  - `stub`

### 10. App wiring
Implemented in:
- `toolforge_env/toolforge_env/server/app.py`

The app uses OpenEnv `create_app(...)`, so the environment is wired in the expected OpenEnv style.

## Current Verified Behavior

Verified examples:
- empty plan short-circuits with score `0.0`
- invalid tool short-circuits with score around `0.2`
- valid plan gets full pipeline result and score `1.0`
- environment `step(valid_action)` returns reward `1.0`
- environment `step(bad_action)` returns reward around `0.2`
- `reset()` still works correctly

## What Is Placeholder Right Now

These parts are intentionally temporary and must be replaced later:

### Slot judge
- no real semantic interpretation
- no real model call
- no real slot-filling reasoning

### Reward calculator
- no partial credit
- no difficulty sensitivity
- no meaningful shaping
- no use of task-specific logic yet

### Token cost calculator
- no real efficiency computation
- no macro savings logic
- no relation to actual plan token usage yet

### Environment progression logic
- no advance-to-next-task behavior
- no retry behavior
- no completion gating
- no macro proposal handling
- no accepted-macro lifecycle

### Simulated user integration
- class exists
- but not properly used to gate behavior in `step()` yet

## Known Design Choices So Far

### Slot mapping
Current temporary mapping includes:
- `restart -> CONFIGURATION_ACTION`

This may be revised later, but the slot library is intentionally being kept stable for now.

### Baseline token cost
Current task `baseline_token_cost` values are derived mechanically from the naive atomic tool sequence for each task.

### Pipeline contract
The environment currently depends only on:
- `final_score`
- `passed_validation`
- `summary`

This is intentional to keep the environment decoupled from the internal judge stages.

## What Is Left To Do

### Judge improvements
- replace placeholder Stage 2 with real semantic judging
- replace placeholder Stage 3 with real reward logic
- replace placeholder Stage 4 with real efficiency logic

### Environment behavior
- integrate simulated-user approval properly
- decide task completion condition
- add task progression
- add retry semantics if needed
- handle macro proposal acceptance/rejection

### Observation and UX
- decide whether detailed judge outputs should be surfaced to the agent
- coordinate observation changes carefully with the environment owner

### Submission and infra
- inference script
- Dockerfile/runtime hardening
- validation and local run workflow
- final README polish
- HF Space deployment path

## Practical Status Summary

Current status can be described as:

- OpenEnv skeleton: done
- task/tool scaffolding: done
- judge pipeline scaffold: done
- validator: real
- reward path wiring: done
- full benchmark logic: not done

This means the codebase now has a stable base and a first layer of real judging behavior, but it is not yet a finished benchmark environment.
