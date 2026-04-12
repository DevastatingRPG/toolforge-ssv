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

# 🚀 ToolForge Adaptive Tool Learning Environment

A deterministic agent optimization environment designed to evaluate and enhance how AI agents learn reusable tool abstractions from repeated user workflows—reducing redundant reasoning, minimizing token usage, and improving execution efficiency over time.

## Quick Start

```python
import asyncio
from client import ToolforgeEnv
from models import ToolForgeAction, ToolCall, Tool

async def main():
    try:
        # Connect to the live HuggingFace Space
        with ToolForgeEnv.from_env("ShubhamSarkar04/toolforge-env") as env:
            # Reset the environment with a specific task tier
            result = await env.reset(task_id="easy")
            obs = result.observation
            
            print("== USER TASK ==")
            print(obs.current_task.prompt)
            print("== AVAILABLE TOOLS ==")
            print(obs.available_tools)

            # Step — submit the agent's plan of action
            action = ToolForgeAction(
                action_type="propose_plan",
                plan=[
                    ToolCall(tool_name="deploy"),
                    ToolCall(tool_name="healthcheck"),
                    ToolCall(tool_name="notify")
                ]
            )
            result = await env.step(action)
            
            print(f"\nScore: {result.reward:.2f}")
            print(f"Feedback: {result.observation.feedback}")

    finally:
        await env.close()
```
## 🧪 Try It Live (Hugging Face Spaces)

> [!TIP]
> A live version of ToolForge is available on Hugging Face Spaces:  
> **https://huggingface.co/spaces/ShubhamSarkar2004/toolforge-env**
>
> 👉 Use the different tabs to explore:
> - **Demo** → Run pre-configured agents on tasks  
> - **Bring Your Own Model** → Plug in your own LLM  
> - **Custom** → Submit your own prompts and compete with the agent  
>
> No setup required — everything runs in-browser.

## 💡 Why This Problem?
Modern tool-enabled AI agents (e.g., using function calling or tool APIs) suffer from a key inefficiency:

- They recompute execution plans from scratch for every request
- They repeatedly select tools, chain calls, and infer dependencies
- They fail to generalize recurring workflows across sessions

This results in:

- **High token consumption**
- **Increased latency**
- **Redundant reasoning cycles**

Even when tasks are structurally identical, agents behave statelessly—treating each request as a new problem.

---

## 🧠 Core Idea

```mermaid
flowchart LR
    A[Repeated Tasks]:::process --> B[Pattern Detection]:::process
    B --> C[Tool Abstraction]:::process
    C --> D[Tool Registry]:::data
    D --> E[Direct Invocation]:::success

    classDef process fill:#3b82f6,color:#fff,stroke:#1e40af
    classDef success fill:#22c55e,color:#fff,stroke:#166534
    classDef data fill:#a855f7,color:#fff,stroke:#6b21a8
```
ToolForge introduces a learning loop where agents evolve their own toolset.

---
## ⚙️ Current Environment Architecture

```mermaid
flowchart TD
    A[Input Provider]:::process --> B[LLM Agent]:::process
    B --> C[Tool Plan Proposal]:::process
    B --> D[Macro Proposal]:::data
    C --> E[Evaluation Engine]:::process
    D --> E
    E --> F{Accepted?}:::decision
    F -- Yes --> G[Persist Macro + Reward]:::success
    F -- No --> H[Reject / Penalize]:::error

    classDef process fill:#3b82f6,color:#fff
    classDef decision fill:#eab308,color:#000
    classDef success fill:#22c55e,color:#fff
    classDef error fill:#ef4444,color:#fff
    classDef data fill:#a855f7,color:#fff
```
## 🔌 Open Integration Design

ToolForge is built to transition from simulation to real-world usage without changing agent logic.

### Key Abstractions

- **InputProvider** → Decouples task source  
  - Today: simulated tasks  
  - Future: human input / APIs / live systems  
  - Enables human-in-the-loop interaction  

  > Ref: [`InputProvider interface`](./server/inputs/base.py)

- **ToolStore** → Decouples tool execution  
  - Today: in-memory simulated tools  
  - Future: real tools via MCP / external APIs  
  - Learned macros behave like real tools  

  > Ref: [`AbstractToolStore`](./server/tools/base.py)

---

### 🚀 What This Enables

- Swap **simulated tasks → real users**
- Swap **mock tools → real systems**
- Reuse learned macros in production

> *Train in simulation. Deploy without rewriting.*

---

## 🔍 Evaluation Pipeline
```mermaid
flowchart TD
    A[Plan + Macro Proposal]:::process --> B[Sanity Validation]:::process
    B -->|Fail| X[Penalty -0.2]:::error
    B -->|Pass| C[Slot Judgment]:::decision

    C -->|Harmful Calls| Y[Min Reward -0.2]:::error
    C -->|Judge Failure| Z[Zero Reward]:::error
    C -->|Valid| D[Reward Feature Extraction]:::process

    D --> E[Scoring Engine]:::process
    E --> F[Final Reward]:::success

    classDef process fill:#3b82f6,color:#fff
    classDef decision fill:#eab308,color:#000
    classDef success fill:#22c55e,color:#fff
    classDef error fill:#ef4444,color:#fff
```
- Strict short-circuiting ensures invalid or unsafe plans are penalized early
- Final reward is bounded between -0.2 and 1.0
---

## 📊 Episode Grading

At the end of each episode, performance is evaluated and assigned a final benchmark score between 0.01 and 0.99 across correctness, efficiency, and macro usage.

### 🧮 Scoring Breakdown

```mermaid
pie title Episode Score Weights
    "Accuracy" : 40
    "Token Optimization" : 30
    "Macro Creation" : 20
    "Macro Usage" : 10
```

### 📈 How It Works

```mermaid
flowchart LR
    A[Episode Metrics]:::process --> B[Sub-scores]:::process
    B --> C[Weighted Sum]:::process
    C --> D[Clamped Score]:::success

    classDef process fill:#3b82f6,color:#fff
    classDef success fill:#22c55e,color:#fff
```
- Accuracy → correctness of plans
- Token Optimization → efficiency on correct plans
- Macro Creation → quality of proposed macros
- Macro Usage → correct reuse of macros

Final score prioritizes correctness first, then efficiency and abstraction.

## What Makes This Different

ToolForge goes beyond simple tool execution benchmarks. It evaluates an agent's ability to abstract and optimize its own capabilities over time, the same challenge faced by any production agentic system operating at scale:

- **Macro Creation:** Agents can submit a `propose_plan_with_macro` action to permanently add a new, cheaper master tool to their environment state.
- **Dual-Objective Scoring:** Plans are evaluated on both **semantic accuracy** (did it solve the task?) and **token efficiency** (did it use macros to save tokens?).
- **Progressive Difficulty:** Tasks range from highly repetitive deployment sprints (easy) to complex, multi-phase infrastructure migrations (hard).

## 📚 Task Curriculums

The environment evaluates agents across **three structured difficulty tiers**, each designed to test progressively advanced capabilities in **tool planning, macro abstraction, and workflow optimization**. Task instances are **dynamically sampled within each curriculum**, preventing memorization and encouraging generalization.

---

| Difficulty | Tier | Curriculum Focus | Active Challenge | Core Competency Evaluated |
|------------|------|------------------|------------------|---------------------------|
| `easy`     | 🟢 Foundational | Repetitive workflows | High-frequency, structurally identical tasks | Identifying recurring tool patterns and proposing reusable macros |
| `medium`   | 🟡 Intermediate | Incident response & operations | Multi-step, goal-driven workflows with branching logic | Adapting plans dynamically while preserving correctness and efficiency |
| `hard`     | 🔴 Advanced | Complex infrastructure workflows | Multi-phase, order-dependent execution chains | Synthesizing long-horizon plans and abstracting reusable high-level macros |

---

## ⚙️ Action & Observation Spaces

### Action: `ToolForgeAction`

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `Literal["propose_plan", "propose_plan_with_macro"]` | Specifies whether the agent submits only a tool execution plan or a plan along with a macro proposal. |
| `plan` | `List[ToolCall]` | Ordered sequence of tool calls representing the agent’s proposed execution strategy for the task. |
| `macro_proposal` | `Optional[Tool]` | Optional composite tool abstraction proposed for reuse. Only applicable when `action_type` is `propose_plan_with_macro`. |

---

### Observation: `ToolForgeObservation`

| Field | Type | Description |
|-------|------|-------------|
| `current_task` | `Task` | The active task containing the prompt, required slots, and baseline expectations. |
| `available_tools` | `List[Dict[str, Any]]` | List of tools and previously accepted macros available to the agent for planning. |
| `grading` | `Optional[EpisodeGradingState]` | Aggregated evaluation signals, including reward breakdown and step-level metrics from the evaluation pipeline. |


## Server Setup & Deployment

### Local Docker (Recommended)

```bash
docker build -t toolforge-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 -e ENABLE_WEB_INTERFACE=true toolforge-env:latest
curl http://localhost:8000/health
```

### Deploy to Hugging Face Spaces

```bash
openenv push --repo-id your-org/toolforge-env
```


## 🛣️ Future Roadmap

### 🔓 Open Integration for Real-World Usage

- The environment is being extended to support **human-in-the-loop prompt interaction** after model training and evaluation on fixed task sets  
- Enables transition from:
  - **Benchmarking environment → Practical agent system**  
- Users will be able to:
  - Submit custom tasks  
  - Leverage learned macros  
  - Evaluate agent efficiency in real workflows  

---

### 🧰 Real Tool Execution via MCP

- Planned integration with **Model Context Protocol (MCP)** to enable:
  - **Actual tool execution** instead of simulated evaluation  
  - Dynamic interaction with external systems (APIs, infra, services)  

- This introduces:
  - Stateful execution environments  
  - Real-world side effects  
  - End-to-end automation capabilities  

---

### 🧠 From Simulation to Execution

```mermaid
flowchart LR
    A[Simulated Evaluation]:::process --> B[Hybrid System]:::decision
    B --> C[Fully Executable Agents]:::success

    classDef process fill:#3b82f6,color:#fff
    classDef decision fill:#eab308,color:#000
    classDef success fill:#22c55e,color:#fff
```
