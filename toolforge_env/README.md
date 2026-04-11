---
title: ToolForge Environment Server
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - agents
  - devops
  - tool-use
  - reinforcement-learning
---

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

## What Makes This Different

ToolForge goes beyond simple tool execution benchmarks. It evaluates an agent's ability to abstract and optimize its own capabilities over time:

- **Macro Creation:** Agents can submit a `propose_plan_with_macro` action to permanently add a new, cheaper master tool to their environment state.
- **Dual-Objective Scoring:** Plans are evaluated on both **semantic accuracy** (did it solve the task?) and **token efficiency** (did it use macros to save tokens?).
- **Progressive Difficulty:** Tasks range from highly repetitive deployment sprints (easy) to complex, multi-phase infrastructure migrations (hard).

## Quick Start

```python
from toolforge_env_environment import ToolforgeEnvironment
from models import ToolForgeAction, ToolCall, Tool

env = ToolforgeEnvironment()
result = env.reset(mode="eval", difficulty="easy")

print(f"Task: {result.current_task.prompt}")

action = ToolForgeAction(
    action_type="propose_plan",
    plan=[
        ToolCall(tool_name="deploy"),
        ToolCall(tool_name="healthcheck"),
        ToolCall(tool_name="notify")
    ]
)
result = env.step(action)
print(f"Reward: {result.reward}")

macro_action = ToolForgeAction(
    action_type="propose_plan_with_macro",
    plan=[
        ToolCall(tool_name="deploy"),
        ToolCall(tool_name="healthcheck"),
        ToolCall(tool_name="notify")
    ],
    macro_proposal=Tool(
        name="deploy_verify_notify",
        description="Standard deployment pipeline",
        is_macro=True,
        token_cost=1, 
        steps=[
            ToolCall(tool_name="deploy"),
            ToolCall(tool_name="healthcheck"),
            ToolCall(tool_name="notify")
        ]
    )
)
result = env.step(macro_action)
```

## Server Setup & Deployment

### Local Docker (Recommended)

```bash
docker build -t toolforge-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 toolforge-env:latest
curl http://localhost:8000/health
```

### Deploy to Hugging Face Spaces

```bash
openenv push --repo-id your-org/toolforge-env
```

## Core API Surface

### Actions

**`ToolForgeAction`**

- `action_type`: `"propose_plan"` or `"propose_plan_with_macro"`
- `plan`: List of `ToolCall`
- `macro_proposal`: Optional `Tool`

### Observations

**`ToolForgeObservation`**

- `current_task`
- `available_tools`
- `reward`
- `done`

## The Toolbox

1. `deploy`
2. `healthcheck`
3. `notify`
4. `rollback`
5. `scale`
6. `restart`

## Evaluation & Rewards

### Reward Bounds

- Step rewards: `-0.2` to `1.0`
- Final score: `0.0` to `1.0`

### 5-Stage Evaluation Pipeline

1. Sanity Validation  
2. Semantic Slot Judgment  
3. Plan Accuracy Score  
4. Token Efficiency  
5. Reward Calculation  

## Task Curriculums

- **Easy**: High-repetition environments designed for quick macro identification and practice.
- **Medium**: Mixed operational goals with lower pattern overlap to test reasoning versus abstraction.
- **Hard**: Complex, multi-phase projects where context must be maintained across long-horizon scenarios.