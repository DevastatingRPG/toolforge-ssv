# ToolForge-Env

A standalone OpenEnv environment simulating a real-world DevOps scenario where an LLM agent must identify recurring tool patterns and compose them into "macro tools".

## Current Scope
This package currently implements the minimal OpenEnv skeleton:
- Pydantic models for Observations, Actions, and tool abstractions.
- FastAPI server setup integrating with `openenv.core`.
- Easy-tier tasks defined with deterministic required steps.
- Reward and grading logic are currently intentionally stubbed.

## Package Layout
- `toolforge_env/toolforge_env/` - The main Python package directory.
  - `models.py` - Core Pydantic definitions.
  - `client.py` - Client wrapper.
  - `server/` - Core environment logic and FastAPI app.

## Running Locally
You can run the environment server locally using the configured script:
```bash
uv run --project . server
```