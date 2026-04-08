import asyncio
import json
import logging
from pprint import pprint
from models import ToolCall, ToolforgeAction, Tool
from server.toolforge_env_environment import ToolforgeEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_available_tools():
    print("\n--- Starting Tools Delivery Test ---\n")
    
    # 1. Initialize environment
    env = ToolforgeEnvironment()
    
    # 2. Reset (Easy Mode)
    obs = env.reset(task_id="easy")
    
    print("\n[RESET COMPLETE]")
    print(f"Task ID: {obs.current_task.id}")
    print(f"Task Prompt: {obs.current_task.prompt}")
    
    # Check tool format
    tools = obs.available_tools
    atomic_tools = [t['name'] for t in tools if not t.get('is_macro', False)]
    macro_tools = [t['name'] for t in tools if t.get('is_macro', False)]
    
    print(f"\nInitial Tools Count: {len(tools)} (Expected ~10+ atomic, 0 macros)")
    print(f"Atomic Tools: {atomic_tools}")
    print(f"Macro Tools: {macro_tools}")
    
    # Basic assertions
    assert len(atomic_tools) > 0, "Missing atomic tools"
    assert len(macro_tools) == 0, "Macros should be empty initially"
    
    # 3. Step 1: Propose a macro
    print("\n\n--- [STEP 1] PROPOSING MACRO ---")
    
    macro_proposal = Tool(
        name="deploy_and_check",
        description="Deploys the service, validates its status and then tests its health endpoint",
        is_macro=True,
        steps=[
            ToolCall(tool_name="deploy"),
            ToolCall(tool_name="healthcheck")
        ]
    )
    
    # Use valid atomic tools for the plan itself
    action = ToolforgeAction(
        action_type="propose_plan_with_macro",
        plan=[
            ToolCall(tool_name="deploy"),
            ToolCall(tool_name="healthcheck")
        ],
        macro_proposal=macro_proposal
    )
    
    obs = env.step(action)
    
    tools = obs.available_tools
    atomic_tools = [t['name'] for t in tools if not t.get('is_macro', False)]
    macro_tools = [t['name'] for t in tools if t.get('is_macro', False)]
    
    print(f"\nPost-Proposal Tools Count: {len(tools)}")
    print(f"Atomic Tools: {atomic_tools}")
    print(f"Macro Tools: {macro_tools}")
    
    # Debug: Did it get added to accepted_macros internally?
    print(f"Internal accepted macros: {[m.name for m in env._state.accepted_macros]}")

    
    # 4. Step 2: Normal step to verify macro persistence
    print("\n\n--- [STEP 2] NORMAL PLAN (Checking persistence) ---")
    action2 = ToolforgeAction(
        action_type="propose_plan",
        plan=[
            ToolCall(tool_name="deploy"),
        ],
    )
    
    obs = env.step(action2)
    tools = obs.available_tools
    atomic_tools = [t['name'] for t in tools if not t.get('is_macro', False)]
    macro_tools = [t['name'] for t in tools if t.get('is_macro', False)]
    
    print(f"\nPost-Step-2 Tools Count: {len(tools)}")
    print(f"Atomic Tools: {atomic_tools}")
    print(f"Macro Tools: {macro_tools}")
    
    print("\n--- Test Complete ---")


if __name__ == "__main__":
    test_available_tools()
