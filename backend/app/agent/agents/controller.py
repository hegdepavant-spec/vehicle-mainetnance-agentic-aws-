"""
ControllerAgent — Root Google ADK LlmAgent orchestrating the multi-agent system.

Coordinates:
- InputAgent (Input LLM: explicit extraction)
- PredictionAgent (ADK wrapper around the authoritative ML tool)
- ResponseAgent (Output LLM: grounded response)
- JudgeAgent (Quality & hallucination evaluation)
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from backend.app.agent.adk_agents.model_bridge import get_primary_model

CONTROLLER_AGENT_INSTRUCTION = """You are ControllerAgent — the root orchestrator of the AutoCare AI multi-agent system.

YOUR ROLE:
You coordinate specialist agents to process the user's message, validate facts, and generate grounded responses.

AGENTS UNDER YOUR ORCHESTRATION:
1. InputAgent: Extracts explicit facts from user message.
2. ResponseAgent: Formats validated state into conversational natural response.
3. JudgeAgent: Evaluates response before sending to user.

WORKFLOW:
1. Delegate extraction to InputAgent.
2. Ensure state updates are validated.
3. Delegate response formatting to ResponseAgent using strictly validated state.
4. Delegate response review to JudgeAgent.
"""


def build_controller_agent() -> LlmAgent:
    """Build and return the root ControllerAgent ADK LlmAgent."""
    from backend.app.agent.agents.input import build_input_agent
    from backend.app.agent.agents.response import build_response_agent
    from backend.app.agent.agents.judge import build_judge_agent
    from backend.app.agent.adk_agents.prediction_agent import build_prediction_agent

    input_agent = build_input_agent()
    response_agent = build_response_agent()
    judge_agent = build_judge_agent()
    prediction_agent = build_prediction_agent()

    return LlmAgent(
        name="ControllerAgent",
        description="Root orchestrator for AutoCare AI multi-agent vehicle maintenance system.",
        model=get_primary_model(),
        instruction=CONTROLLER_AGENT_INSTRUCTION,
        tools=[
            AgentTool(agent=input_agent),
            AgentTool(agent=response_agent),
            AgentTool(agent=judge_agent),
            AgentTool(agent=prediction_agent),
        ],
    )
