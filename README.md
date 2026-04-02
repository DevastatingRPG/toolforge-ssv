# ToolForge-Env

ToolForge-Env is an OpenEnv-compatible benchmark for studying how an agent learns repeated workflow structure across tasks, proposes reusable macro tools, and reduces overall interaction cost over an episode.

## Problem Statement

The initial motivating example comes from DevOps-style workflow automation: an agent repeatedly receives operational tasks, solves them by composing atomic tools, notices repeated tool-call patterns, and proposes a new macro tool that can be reused later. The broader benchmark direction is more general than DevOps: the same environment pattern can support any domain where tasks are solved through repeated structured tool use.

## Current Scope

This repository currently contains the base OpenEnv skeleton for the environment:

- Typed Pydantic models for actions, observations, state, tools, tasks, and macro proposals
- A standalone OpenEnv package layout
- A server wired through `openenv.core.env_server.create_app(...)`
- A deterministic easy-task task bank
- A simulated user stub for plan evaluation
- A deterministic stub `step()` implementation that updates accounting fields

The following parts are intentionally still stubbed or incomplete:

- reward shaping
- task graders
- medium and hard tasks
- macro approval logic
- task progression across the episode
- baseline inference script
- Docker and deployment hardening

## Environment Idea

Each episode presents a sequence of tasks. The agent must:

1. Read the current task and available tool definitions
2. Submit a structured plan composed of tool calls
3. Optionally propose a macro tool built from repeated call patterns
4. Reuse previously accepted macros in later tasks
5. Maximize task success while minimizing cumulative interaction cost

The benchmark is designed for environments where:

- tasks are naturally expressed as multi-step tool workflows
- repeated tool-call subsequences appear across tasks
- abstraction and reuse should improve efficiency

## Project Layout

The current standalone package layout is:

```text
openenv-SSV/
  openenv.yaml
  pyproject.toml
  README.md
  toolforge_env/
    toolforge_env/
      __init__.py
      models.py
      client.py
      server/
        __init__.py
        app.py
        toolforge_environment.py
        tasks.py
        tools.py
        user_sim.py
```

## OpenEnv Conventions

This project follows the standard OpenEnv environment pattern:

- typed `Action`, `Observation`, and `State` models
- server-side `Environment` implementation with `reset()`, `step()`, and `state`
- app wiring through `create_app(...)`
- package metadata in `openenv.yaml`

The implementation style is based on the OpenEnv tutorials, template environment, and reference environments such as `coding_env`.

## Current Runtime Behavior

At the current stage:

- `reset()` initializes a deterministic episode state and returns the first observation
- `state` returns the full current environment state
- `step()` is a stubbed transition that records the submitted plan, updates token accounting, and returns a valid observation

This means the environment is structurally usable for early integration and debugging, but it is not yet a full benchmark implementation.

## Local Development

Install the package from the project root:

```bash
pip install -e .
```

Run the server locally:

```bash
python -m uvicorn toolforge_env.server.app:app --host 0.0.0.0 --port 8000
```

Once running, the standard OpenEnv endpoints should be available, including:

- `/health`
- `/reset`
- `/step`
- `/state`
- `/schema`
- `/ws`

## Next Implementation Areas

The next major milestones are:

- finalize the episode/task progression logic
- implement deterministic grading and reward shaping
- define macro proposal validation and acceptance
- add a baseline inference script
- prepare Docker and Hugging Face Space deployment
