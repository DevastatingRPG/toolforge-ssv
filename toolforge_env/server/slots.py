"""Semantic slot definitions for the ToolForge judge.

Each slot represents a high-level *intent* that a tool call can fulfil
within a DevOps workflow. The judge maps concrete tool calls to these
abstract slots when evaluating whether a plan satisfies a task.

This file is pure constants — no functions, no side effects.
"""

# Maps slot name → natural-language description of the intent.
# The description is used by the Stage-2 semantic judge to determine
# whether a given tool call fills the slot.
DEVOPS_SLOTS = {
    "DEPLOYMENT_ACTION": (
        "An action that deploys, releases, pushes, or ships a new version "
        "of a service into a target environment."
    ),
    "VERIFICATION_ACTION": (
        "An action that checks, tests, pings, or validates that a service "
        "is healthy and operating correctly."
    ),
    "NOTIFICATION_ACTION": (
        "An action that informs, alerts, or notifies a person or channel "
        "about the outcome of an operation."
    ),
    "ROLLBACK_ACTION": (
        "An action that reverts or restores a service to a previous stable "
        "version after a failed change."
    ),
    "SCALING_ACTION": (
        "An action that increases or decreases the capacity of a service "
        "by adjusting its replica count or resource allocation."
    ),
    "CONFIGURATION_ACTION": (
        "An action that modifies the runtime configuration of a service, "
        "such as restarting it, toggling feature flags, or updating "
        "environment variables."
    ),
}
