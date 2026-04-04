Here is the report on the judge code consolidation pass:

### 1. Files Created
- `toolforge_env/server/plan_evaluator.py`
- `toolforge_env/server/evaluation_pipeline.py`
- `toolforge_env/server/test_evaluation_pipeline.py`

### 2. Files Deleted
- `toolforge_env/server/user_sim.py`
- `toolforge_env/server/validator.py`
- `toolforge_env/server/slot_judge.py`
- `toolforge_env/server/reward_calculator.py`
- `toolforge_env/server/token_calculator.py`
- `toolforge_env/server/judge_pipeline.py`

### 3. Files Modified
- None of the core OpenEnv server or environment files were modified (`toolforge_environment.py`, `app.py`, `client.py`, etc.). The modifications were strictly limited to creating the new judge files and deleting the old logic files, fulfilling the clean layout constraint.

### 4. New `plan_evaluator.py` Public APIs
The evaluation logic is now consolidated as discrete, pure helper functions within a single file. The exposed interfaces are:
- `def get_relevant_slots(required_slots: List[str]) -> Dict[str, str]`
- `def run_sanity_validation(plan: List[ToolCall], available_tools: Dict[str, Tool]) -> ValidationResult`
- `def run_slot_judgment(task_prompt: str, required_slots: List[str], slot_definitions: Dict[str, str], available_tools: List[Tool], plan: List[ToolCall]) -> SlotJudgmentResult`
- `def run_reward_calculation(validation_result: ValidationResult, slot_judgment: SlotJudgmentResult, task: Task, goal_achieved: bool = False) -> RewardResult`
- `def run_token_calculation(plan: List[ToolCall], accepted_macros: List[Tool], baseline_token_cost: int) -> TokenCostResult`
- `def compute_final_score(reward: RewardResult, token_cost: TokenCostResult) -> float`

### 5. New `run_evaluation_pipeline(...)` Signature
The orchestration layer (`evaluation_pipeline.py`) exposes a single thin public wrapper:
```python
def run_evaluation_pipeline(
    plan: List[ToolCall],
    task: Task,
    available_tools: Dict[str, Tool],
    accepted_macros: List[Tool],
    baseline_token_cost: int,
    goal_achieved: bool = False,
) -> PipelineResult
```
This wrapper manages the flow exactly as before, with a short-circuit on structural validation failures and a new `goal_achieved` hook.

### 6. What remains placeholder and what is real
**Real:**
- Structural Stage-1 validation (`run_sanity_validation`) correctly enforces empty plans, required keys, extra keys, and correct tool scope according to the tool JSON schmea.
- Pipeline orchestration and short-circuit control flow fully evaluates validity flags.
- Typed Result Models natively map data up the chain consistently.

**Placeholder:**
- Stage-2 Slot assignment (`run_slot_judgment`) uses a private placeholder array to return correctly shaped mock evaluations that say the semantic task is complete, without making an LLM call.
- Stage-3 Reward and Stage-4 Token Efficiency logic (`run_reward_calculation` & `run_token_calculation`) emit deterministic constants instead of mathematically penalizing behavior, though `run_reward_calculation` is wired fully to accept a goal achievement bonus in preparation for real shaping.
