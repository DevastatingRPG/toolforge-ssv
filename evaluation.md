# Consolidated Evaluation System — Iteration 1

This document outlines the implementation details of the first real iteration of the ToolForge evaluation system, guided by the philosophy of mathematical scoring.

## 1. Files Changed
- **Modified:** `toolforge_env/models.py` (Replaced `RewardResult` with `PlanAccuracyResult`, updated inner details for `ToolEvaluation`, `SlotJudgmentResult`, `TokenCostResult`, and `PipelineResult`).
- **Modified:** `toolforge_env/server/plan_evaluator.py` (Completely implemented the semantic Stage 2 simulation and mathematical curves for Stages 3 and 4).
- **Modified:** `toolforge_env/server/evaluation_pipeline.py` (Implemented the new orchestrator method flow with explicit structural and harmful short-circuits).
- **Modified:** `toolforge_env/server/test_evaluation_pipeline.py` (Added test suites reflecting the Stage 3 negative bounds and macro-efficiency bonuses).

*Note: The environment wiring files (`toolforge_environment.py`, `app.py`, `client.py`) were left completely untouched per constraints. Testing via run instructions currently throws a `ModuleNotFoundError` due to the environment holding onto the retired `judge_pipeline` import, which the environment owner will update on their side.*

## 2. Exposed Public APIs (`plan_evaluator.py`)
The evaluation logic remains the single source of truth entirely contained in `plan_evaluator.py`. The exposed public interfaces are:
- `get_relevant_slots(required_slots: List[str]) -> Dict[str, str]`
- `run_sanity_validation(plan: List[ToolCall], available_tools: Dict[str, Tool]) -> ValidationResult`
- `run_slot_judgment(...) -> SlotJudgmentResult` (with simulated LLM relevance classifications)
- `plan_accuracy_score(validation_result: ValidationResult, slot_judgment: SlotJudgmentResult, task: Task, goal_achieved: bool = False) -> PlanAccuracyResult`
- `run_token_calculation(plan: List[ToolCall], accepted_macros: List[Tool], task: Task, available_tools: Dict[str, Tool]) -> TokenCostResult`
- `reward_calculation(plan_accuracy: PlanAccuracyResult, token_cost: TokenCostResult) -> float`

## 3. Renamed Methods
The orchestrator correctly sequences the new method names:
- `run_reward_calculation` -> `plan_accuracy_score`
- `compute_final_score` -> `reward_calculation`

## 4. Model Field Changes
The OpenEnv data models correctly trace data upward across the five stages:
- **`ToolEvaluation`**: Replaced Boolean `correct` with the explicit `classification: Literal["relevant", "unnecessary", "harmful"]`.
- **`SlotJudgmentResult`**: Added `harmful_calls_present: bool`.
- **`PlanAccuracyResult`**: Entirely replaced `RewardResult` to encapsulate Stage 3 details: `slot_completion_ratio`, `slot_score`, `unnecessary_penalty`, and `score` (bounded strictly to `[-1.0, 0.0]`).
- **`TokenCostResult`**: Added `macro_bonus: float` for Stage 4 efficiency bumps.
- **`PipelineResult`**: Replaced `reward` field with `plan_accuracy: Optional[PlanAccuracyResult]`.

## 5. Iteration 1 Constants
Configurable thresholds and scaling factors were grouped to the top of `plan_evaluator.py`:
- `PLAN_ACCURACY_WEIGHT = 0.7`
- `TOKEN_EFFICIENCY_WEIGHT = 0.3`
- `SLOT_COMPLETION_CURVE_K = 3.0`
- `UNNECESSARY_CALL_PENALTY = 0.05`
- `MAX_UNNECESSARY_CALL_PENALTY = 0.20`
- `MACRO_USAGE_BONUS = 0.10`

## 6. Simulated vs. Real Logic

### Simulated
- **LLM Semantic Judge**: `run_slot_judgment` operates locally. It runs a deterministic rule-based simulation identifying required slots mapped sequentially, marks unknown extras as "unnecessary", and specifically traps tools featuring the string `"delete"` or `"drop"` as "harmful".

### Real
- **Mathematical Curves**: Stage 3 logic processes true reverse-exponential curves ensuring a perfect atomic plan registers specifically as `0.0`, with mistakes dragging the reward deeply negative.
- **Token Efficiency**: Stage 4 dynamically registers base token costs against known constraints and provides `macro_bonus` values only when tokens are truly saved by macro utilization.
- **Structural Pipeline Flow**: The evaluation orchestrator properly parses returns and short-circuits directly on structural invalidity or simulated semantic hazard detection (`harmful`), entirely mirroring intended final logic.
