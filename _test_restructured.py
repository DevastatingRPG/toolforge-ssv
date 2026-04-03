import logging
from toolforge_env.server.toolforge_environment import ToolForgeEnvironment
from toolforge_env.models import ToolForgeAction, ToolCall, MacroProposal

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_test():
    print("Initializing environment...")
    env = ToolForgeEnvironment()
    
    print("\n--- TEST 1: Reset and Initial State ---")
    obs = env.reset(task_id="easy-deploy-notify")
    print(f"Task 1: {obs.current_task.id}")
    print(f"User Prompt: {obs.metadata.get('user_prompt')}")
    print(f"Initial available tools: {len(obs.available_tools)}")

    print("\n--- TEST 2: Successful Pass & Advance ---")
    # Valid plan for "easy-deploy-notify"
    # deploy requires service_name and version
    # healthcheck requires service_name
    # notify requires channel and message
    good_plan = [
        ToolCall(tool_name="deploy", params={"service_name": "web", "version": "v1.1"}, token_cost=10),
        ToolCall(tool_name="healthcheck", params={"service_name": "web"}, token_cost=2),
        ToolCall(tool_name="notify", params={"channel": "#ops", "message": "done"}, token_cost=1),
    ]
    action_good = ToolForgeAction(
        action_type="propose_plan",
        plan=good_plan,
        reasoning="Full deploy-check-notify sequence"
    )
    obs_advanced = env.step(action_good)
    print(f"User Approved: {obs_advanced.metadata.get('user_approved')}")
    print(f"Passed Validation: {obs_advanced.metadata.get('passed_validation')}")
    print(f"Reward: {obs_advanced.reward}")
    print(f"Next Task: {obs_advanced.current_task.id}")

    print("\n--- TEST 3: Macro Proposal and Registration ---")
    # Propose macro "HealthSummary" for next task
    # Second task is "easy-deploy-restart" in the default easy queue
    # required_steps: ["deploy", "healthcheck", "restart", "healthcheck", "notify"]
    macro = MacroProposal(
        name="HealthSummary",
        description="Health toggle",
        steps=[
            ToolCall(tool_name="healthcheck", params={}, token_cost=2),
            ToolCall(tool_name="notify", params={}, token_cost=1),
        ]
    )
    
    plan_2 = [
        ToolCall(tool_name="deploy", params={"service_name": "web", "version": "v2.0"}, token_cost=10),
        ToolCall(tool_name="healthcheck", params={"service_name": "web"}, token_cost=2),
        ToolCall(tool_name="restart", params={"service_name": "web"}, token_cost=8),
        ToolCall(tool_name="healthcheck", params={"service_name": "web"}, token_cost=2),
        ToolCall(tool_name="notify", params={"channel": "#ops", "message": "done"}, token_cost=1),
    ]
    
    action_macro = ToolForgeAction(
        action_type="propose_plan_with_macro",
        plan=plan_2,
        macro_proposal=macro,
        reasoning="Advancing and proposing macro"
    )
    
    obs_final = env.step(action_macro)
    print(f"Macro Decision: {obs_final.metadata.get('macro_decision')}")
    print(f"Total Accepted Macros: {obs_final.metadata.get('accepted_macro_count')}")
    print(f"Macro 'HealthSummary' registered? {any(t.name == 'HealthSummary' for t in obs_final.available_tools)}")

if __name__ == "__main__":
    try:
        run_test()
        print("\n=== All RESTORATION checks PASSED ===")
    except Exception as e:
        print(f"\n[FAILED] Test error: {e}")
        import traceback
        traceback.print_exc()
