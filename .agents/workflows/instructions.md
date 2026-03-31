---
description: Core agent instructions, tracking mechanisms, and coding conventions
---

# Agent Instructions

For every run, the agent must carefully adhere to these rules and conventions:

1. **Handling Missing Information (Tracking Mechanism):**
   - While coding, if you feel that some details are missing in the prompt or plan, you must write what information is missing as a comment at the **very top** of the edited file.
   - Once the user provides that missing information in a subsequent prompt, you must remove that comment line from the file. This acts as our tracking mechanism for partial vs. completed implementations.
   - Never touch untaught details without explicitly asking the user first.

2. **Workspace & Reference Protocol:**
   - The main workspace and project directory is `openenv-SSV`.
   - The `OpenEnv` folder (e.g., `openenv-course/OpenEnv`) serves as the tutorial and helper method reference. Consult it for API standards, examples, and OpenEnv-specific implementations.

3. **Self-Updating Instructions:**
   - If the user ever gives a new general instruction, rule, or preference, you must automatically add it to this file (`.agents/workflows/instructions.md`). This ensures instructions persist across subsequent runs.

4. **Coding Conventions:**
   - Write clean, well-documented, and modular code.
   - Do not modify files or boilerplate outside the scope of the user's explicit instructions.
   - Provide type hints for Python function signatures, and follow standard styling (e.g., PEP 8).
   - Variables, classes, and functions should have descriptive and meaningful names.
   - **Logging**: Code must include proper logging (e.g., `import logging`).
   - **Commenting Requirements**: Code must be heavily commented.
     - **Classes**: Each class must have a comment/docstring describing what the class is for.
     - **Methods**: Each method should have "butterfly comments" (decorative/prominent block comments) clearly outlining its purpose.
     - **Variables**: Each variable, whether global or a class variable, must have an accompanying comment explaining what it's used for.
   - Any future coding conventions provided by the user must be appended to this list.

---

# Hackathon Guidelines & Constraints

These are the strict requirements for the Meta & HuggingFace AI Hackathon. The environment we build (ToolForge-Env) must strictly align with these at all times.

## 1. Functional Requirements
- **Real-world task simulation**: Must simulate actual tasks (e.g., email triage, code review, data cleaning, DevOps workflows), NOT games or toys.
- **OpenEnv Spec Compliance**:
  - Implement full OpenEnv interface: typed Observation, Action, Reward Pydantic models.
  - Endpoints: `step(action) -> (observation, reward, done, info)`, `reset() -> observation`, `state() -> state`.
  - Must include `openenv.yaml` with metadata.
  - Must pass `openenv validate`.
- **Minimum 3 tasks with agent graders**:
  - Concrete objectives with a difficulty range: easy -> medium -> hard.
  - Graders must be programmatic, scoring between 0.0 and 1.0 reliably, with deterministic success/failure criteria.
- **Meaningful reward function**:
  - Provide continuous signal over the trajectory, rewarding partial progress.
  - Penalize undesirable behaviors (e.g., loops, destructive actions, excessive token usage).
- **Baseline inference script**:
  - Must use the `openai` Python client.
  - Must read API credentials from exact environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
  - The script *must* be named `inference.py` and placed in the project root.
  - Produces reproducible baseline scores across all 3 tasks.
  - Script runtime must be < 20 minutes.

## 2. Non-Functional Requirements & Infrastructure Restrictions
- **Deploys to Hugging Face Space**: Must run as a containerized HF Space, tagged with `openenv`.
- **Containerized Execution**: Must include a working `Dockerfile`. Starts cleanly with `docker build` & `docker run`.
- **Resource Limits**: Environment and inference script must be capable of running on a machine with vCPU=2 and Memory=8GB.
- **Documentation (`README.md`)**: Must include:
  - Environment description and motivation.
  - Action and observation space definitions.
  - Task descriptions with expected difficulty.
  - Setup and usage instructions.
  - Baseline scores.

## 3. Pre-Submission Checklist & Validation
A supplied `prevalidation_script.sh` requires:
- **HF Space deploys**: Automated ping to Space URL returns HTTP 200 on `/reset`.
- **OpenEnv Spec compliance**: `openenv validate` passes.
- **Dockerfile builds**: Successful image build from root or `server/` directory.
- **Baseline Reproduces**: The `inference.py` script runs without error.
- **3+ Tasks with Graders**: Valid scores returned between 0.0 and 1.0.

## 4. Evaluation Criteria Overview
- **Real-world utility (30%)**: Solves a valid domain modeling gap (e.g., Devops / Tool workflow reduction).
- **Task & grader quality (25%)**: 3+ tasks with varying difficulty, deterministic 0.0-1.0 grading.
- **Environment design (20%)**: Clean states, sensible Pydantic models, non-sparse reward shaping, proper episode boundaries.
- **Code quality & spec compliance (15%)**: `openenv validate`, working Dockerfile, documented, tested, baseline completion.
- **Creativity & novelty (10%)**: Novel domain, interesting mechanics. (Our approach of discovering macro-tools fits perfectly).
