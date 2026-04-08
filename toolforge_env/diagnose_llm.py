import os
import json
import sys

# Ensure the correct path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load .env file manually
def load_env():
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip('"').strip("'")
        print("Loaded environment variables from .env")
    else:
        print(".env file not found")

load_env()

# Import after setting env vars
try:
    from server.evaluation.plan_evaluator import (
        _call_llm_slot_judgment, 
        run_slot_judgment,
        API_BASE_URL, 
        MODEL_NAME, 
        HF_TOKEN
    )
    from models import Tool, ToolCall
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_full_pipeline_parser():
    print(f"--- LLM Parser Diagnostic ---")
    print(f"Target URL: {API_BASE_URL}")
    print(f"Model ID:   {MODEL_NAME}")
    
    # Mock data
    task_prompt = "Deploy a simple web app and check its health."
    required_slots = ["deployment_execution", "health_check"]
    slot_definitions = {
        "deployment_execution": "The act of deploying the primary application or service.",
        "health_check": "Verifying that the application is running correctly after deployment."
    }
    available_tools = {
        "deploy": Tool(name="deploy", description="Deploys the app"),
        "healthcheck": Tool(name="healthcheck", description="Checks health")
    }
    plan = [
        ToolCall(tool_name="deploy"),
        ToolCall(tool_name="healthcheck")
    ]
    
    print("\n1. Testing raw _call_llm_slot_judgment...")
    try:
        raw_json = _call_llm_slot_judgment(
            task_prompt=task_prompt,
            required_slots=required_slots,
            slot_definitions=slot_definitions,
            available_tools=list(available_tools.values()),
            plan=plan
        )
        print("SUCCESS: Raw LLM response received.")
        print(json.dumps(raw_json, indent=2))
    except Exception as e:
        print(f"FAILED: {e}")
        return

    print("\n2. Testing integrated run_slot_judgment (Parser Test)...")
    try:
        # Note: run_slot_judgment expects available_tools as a List[Tool]
        result = run_slot_judgment(
            task_prompt=task_prompt,
            required_slots=required_slots,
            slot_definitions=slot_definitions,
            available_tools=list(available_tools.values()),
            plan=plan
        )
        print("SUCCESS: Parser integrated correctly.")
        print(f"  Slots Filled: {result.slots_filled}")
        print(f"  Slots Missing: {result.slots_missing}")
        print(f"  Harmful Detected: {result.harmful_calls_present}")
        print(f"  Task Complete: {result.task_complete}")
        
    except Exception as e:
        print(f"FAILED Parser Error: {e}")

if __name__ == "__main__":
    test_full_pipeline_parser()
