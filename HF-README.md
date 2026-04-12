# 🔨 ToolForge — Adaptive Tool Learning for Agentic AI

ToolForge is a reinforcement learning environment that trains AI agents to stop re-solving 
workflows they have already learned. Instead of reasoning from scratch on every tool-calling 
task, agents learn to compress recurring multi-step sequences — deploy, verify, notify — into 
reusable macro tools, cutting inference turns and token cost on repeated workflows. Unlike 
standard tool-use benchmarks that only measure task completion, ToolForge scores agents on 
both correctness and efficiency together, so abstraction is only rewarded when it is built on 
top of a correct plan. Bring your own agent via API key or local model, watch it learn across 
episodes, and export the full reward signal for your own training loop.