# ToolForge Integration Guide

This document is for integrating the current judging pipeline work with the main environment implementation.

## Purpose

The current codebase now has a working skeleton for:
- typed environment models
- task definitions
- tool definitions
- a judge pipeline with 4 stages
- environment-side pipeline invocation

The goal of this guide is to clarify:
- what is already implemented
- what is placeholder logic
- what outputs are currently stable
- what the environment creator needs to integrate next

## Current Integration Boundary

The intended integration boundary is:

- The environment should call only:
  - `run_judge_pipeline(...)`
- The environment should not directly call:
  - `validate_plan(...)`
  - `judge_plan(...)`
  - `calculate_reward(...)`
  - `calculate_token_cost(...)`

This keeps the environment thin and makes the judging system modular.

## Files Relevant to Environment Integration

### Core models
- `toolforge_env/toolforge_env/models.py`

Contains:
- `ToolCall`
- `Tool`
- `Task`
- `MacroProposal`
- `ToolForgeAction`
- `ToolForgeObservation`
- `ToolForgeState`

Judge result models now also live here:
- `ValidationResult`
- `ToolEvaluation`
- `SlotJudgmentResult`
- `RewardResult`
- `TokenCostResult`
- `PipelineResult`

### Tool definitions
- `toolforge_env/toolforge_env/server/tools.py`

Provides atomic tools and their `params_schema`.
These schemas are used directly by Stage 1 validation.

### Task definitions
- `toolforge_env/toolforge_env/server/tasks.py`

Each task now includes:
- `required_steps`
- `core_steps`
- `required_slots`
- `baseline_token_cost`

These are important because:
- `required_slots` feeds the semantic judge
- `baseline_token_cost` feeds the token cost stage

### Slot library
- `toolforge_env/toolforge_env/server/slots.py`

Contains:
- `DEVOPS_SLOTS`

This is the central slot-definition library used by the pipeline helper.

### Judge stages
- `toolforge_env/toolforge_env/server/validator.py`
- `toolforge_env/toolforge_env/server/slot_judge.py`
- `toolforge_env/toolforge_env/server/reward_calculator.py`
- `toolforge_env/toolforge_env/server/token_calculator.py`

### Orchestrator
- `toolforge_env/toolforge_env/server/judge_pipeline.py`

This is the only judge entrypoint the environment should use.

### Environment
- `toolforge_env/toolforge_env/server/toolforge_environment.py`

This already calls the pipeline in `step()`.

## What Is Fully Implemented

### 1. Structural validation
`validate_plan(...)` is real, deterministic code.

It checks:
- empty plan
- invalid tool name
- missing required params
- extra unknown params

It reads parameter requirements from the tool schema:
- required params from `params_schema["required"]`
- allowed params from `params_schema["properties"]`

This stage is not placeholder.

### 2. Pipeline short-circuit behavior
`run_judge_pipeline(...)` already short-circuits correctly when validation fails.

Current behavior:
- `slot_judgment = None`
- `reward = None`
- `token_cost = None`
- `passed_validation = False`
- `summary = "Validation failed: <reason>"`
- `final_score = max(0.0, 1.0 + validation.penalty)`

This behavior is real and stable.

### 3. Environment integration path
`ToolForgeEnvironment.step()` already:
- increments `step_count`
- appends tool calls to `call_history`
- accumulates `tokens_used`
- calls `run_judge_pipeline(...)`
- sets `obs.reward = pipeline_result.final_score`
- writes lightweight metadata:
  - `summary`
  - `passed_validation`
  - `stub`

This is currently the active integration path.

## What Is Placeholder Right Now

### Stage 2 — semantic slot judge
File:
- `toolforge_env/toolforge_env/server/slot_judge.py`

Current behavior:
- always returns a success-shaped result
- marks all required slots as filled
- returns `task_complete = True`

Important:
- no real OpenAI call yet
- no real semantic reasoning yet

### Stage 3 — reward calculator
File:
- `toolforge_env/toolforge_env/server/reward_calculator.py`

Current behavior:
- fixed placeholder values
- always returns full correctness

### Stage 4 — token cost calculator
File:
- `toolforge_env/toolforge_env/server/token_calculator.py`

Current behavior:
- fixed placeholder values
- does not compute real token usage efficiency yet

## Dynamic Outputs vs Placeholder Outputs

### Dynamic outputs already available
These outputs change based on real input:
- validation pass/fail
- validation reason
- validation penalty
- pipeline short-circuit path
- final score for invalid plans
- `passed_validation`
- environment `step_count`
- environment `call_history`
- environment `tokens_used`

### Placeholder outputs for now
These currently do not reflect real task correctness:
- semantic slot evaluation details
- slot filling completeness
- reward breakdown
- token efficiency
- final score for structurally valid plans
  - currently tends to become `1.0` because placeholder stages are perfect

## Important Current Behavior

### Valid plan
A structurally valid plan currently gets:
- `passed_validation = True`
- placeholder semantic success
- placeholder reward success
- placeholder token success
- `final_score = 1.0`

### Invalid plan
An invalid plan currently gets:
- `passed_validation = False`
- no semantic stage
- no reward stage
- no token stage
- score based only on validation penalty

## What The Environment Creator Should Integrate Next

### 1. Simulated user decision point
Right now `step()` runs the judge pipeline directly after accounting.

Planned future integration:
- simulated user approval/rejection should happen before pipeline execution
- if rejected, environment can keep the task active without pipeline success flow
- this logic is not implemented yet

### 2. Task progression
Not implemented yet.

The environment creator will eventually need to decide:
- when a task is considered complete
- when to move to next task
- when to keep retrying current task

For now:
- do not build this on top of placeholder semantic success
- wait until reward/slot semantics are real or use a temporary explicit condition

### 3. Macro acceptance flow
Not implemented yet.

Currently:
- accepted macros exist in state
- macro proposals exist in action shape
- but no acceptance logic is wired

### 4. Observation exposure
Current observation does not expose detailed judge outputs directly.

Only lightweight metadata is surfaced:
- `summary`
- `passed_validation`
- `stub`

If richer judge information is later needed in observation, coordinate carefully because observation design is still shared work.

## Recommended Integration Rule

For now, treat these three pipeline outputs as the stable contract:
- `pipeline_result.final_score`
- `pipeline_result.passed_validation`
- `pipeline_result.summary`

Avoid depending on deeper placeholder internals for environment flow until the real judge logic lands.

## Current Slot Mapping Note

Current task-slot mapping includes:
- `restart -> CONFIGURATION_ACTION`

This is temporary and accepted for now.
Do not introduce a separate `RESTART_ACTION` unless the team explicitly decides to change the slot taxonomy later.

## Recommended Next Integration Steps

1. Keep environment using only `run_judge_pipeline(...)`
2. Add simulated-user gate before pipeline, if that work begins
3. Delay task-advance logic until reward/semantic logic is less placeholder-heavy
4. Keep observation changes minimal unless explicitly coordinated
5. Use `summary` and `passed_validation` for debugging while the pipeline matures
