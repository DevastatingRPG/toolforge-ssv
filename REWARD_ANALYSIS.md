# Reward Analysis Report

This report summarizes the results from the evaluation pipeline test run logged on 2026-04-07. It analyzes how rewards are assigned across different scenarios, focusing on semantic completion, macro usage, and validation penalties.

## Episode Summary Table

| Episode | Step | Scenario | Task ID | Reward | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Macro Maturity** | 1 | repeat-sequence-seed-1 | easy-deploy-notify | **-0.665** | Incomplete (2/3 slots) |
| | 2 | repeat-sequence-seed-2 | easy-deploy-restart | **-0.878** | Incomplete (2/5 slots) |
| | 3 | mature-macro-creation | easy-deploy-scale | **-0.818** | Incomplete, Macro Approved |
| **B: Premature Macro** | 1 | premature-macro-creation | easy-deploy-notify | **-0.665** | Incomplete, Macro Approved |
| | 2 | plan-with-macro-call | easy-deploy-restart | **-0.957** | Incomplete (1/5 slots) |
| | 3 | plan-with-macro-atomic | easy-deploy-scale | **0.060** | **Complete (Macro Bonus)** |
| **C: Path Validation** | 1 | wrong-tool-call | easy-deploy-notify | **-0.800** | Invalid Tool Penalty |
| | 2 | zero-filled-slots | easy-deploy-restart | **-0.957** | Incomplete (1/5 slots) |
| | 3 | partially-filled-slots | easy-deploy-scale | **-0.941** | Incomplete (1/4 slots) |
| **D: Failure Paths** | 1 | all-filled-slots | easy-deploy-notify | **0.000** | **Complete (Parity)** |
| | 2 | wrong-macro-creation | easy-deploy-restart | **-0.878** | Incomplete, Macro Rejected |
| | 3 | wrong-macro-call | easy-deploy-scale | **-0.800** | Invalid Tool Penalty |

## Reward Extremes

### 🏆 Max Positive Reward: `0.060`
Observed in **Episode B Step 3**.
- **Context:** The plan is semantically complete and correctly uses an approved macro.
- **Reason:** Even though the total call count matches the human baseline (Parity), the agent receives a "Learning Bonus" for using a compressed representation (macro). This rewards the agent for identifying and applying recurring patterns, even before they lead to a net reduction in call count compared to the baseline. Atomic parity without macros (as in Episode D Step 1) still results in exactly 0.0.

### 💀 Max Negative Reward: `-0.957`
Observed in **Episode B Step 2** and **Episode C Step 2**.
- **Context:** Only 1 out of 5 required slots were filled (20% completion).
- **Reason:** The semantic accuracy uses an exponential decay curve. Completing only 20% of a complex task triggers a heavy penalty to discourage minimal efforts.

### ⚠️ Validation Penalty: `-0.800`
Observed in **Episode C Step 1** and **Episode D Step 3**.
- **Context:** The plan contained an `INVALID_TOOL` name.
- **Reason:** Structural errors trigger a flat short-circuit penalty of -0.8 to prioritize valid syntax before semantic logic.

## Future Improvements
- **Efficiency Bonus:** Positive rewards > 0.06 can be achieved by using macros that significantly reduce the plan length below the human baseline.
- **Macro Descriptions:** As noted in the main README, adding descriptions to macros will allow the LLM to judge them directly without internal expansion, potentially refining the "Utility Bonus" logic.
