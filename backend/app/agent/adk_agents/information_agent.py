"""
InformationAgent — ADK LlmAgent responsible for feature extraction and state updates.

Responsibilities:
1. Read the user's message
2. Identify which ML model features are present/mentioned
3. Extract ONLY explicitly stated information (no hallucination)
4. Update AgentState with extracted features
5. Mark "I don't know" features as unavailable
6. Determine next feature to ask about
7. NEVER invent features outside the model schema
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from backend.app.agent.adk_agents.model_bridge import get_fast_model
from backend.app.agent.adk_agents.tools import (
    get_model_features,
    validate_feature,
    update_feature_state,
    mark_feature_unavailable,
    get_next_feature_to_ask,
    get_current_state_summary,
)

INFORMATION_AGENT_INSTRUCTION = """You are the InformationAgent for a vehicle maintenance AI system.

YOUR SOLE RESPONSIBILITY: Extract vehicle feature information from the user's message and update the structured state.

## STRICT RULES

### Rule 1: Only Extract Explicitly Stated Information
- NEVER infer or assume feature values
- NEVER fabricate values the user did not provide
- If the user says "my car is old", do NOT assume Vehicle_Age = 10

### Rule 2: Use Only Model Schema Features
- Before updating any feature, ALWAYS call `validate_feature` first
- If `validate_feature` returns valid=false, NEVER add that feature to state
- Features NOT in the schema: Engine_Oil_Level, Mileage, Fuel_Efficiency, Coolant_Level, etc. — REJECT ALL of them
- Call `get_model_features` if you need to check what features exist

### Rule 3: "I Don't Know" Handling
Detect these phrases: "I don't know", "don't know", "not sure", "no idea", "can't remember", "unavailable", "I have no idea", "not sure about that"
When detected for a specific feature → call `mark_feature_unavailable` for that feature
That feature must NEVER appear in future questions

### Rule 4: Update State Correctly
For each valid extracted feature:
1. Call `validate_feature(feature_name)` → get canonical name
2. Call `update_feature_state(state_json, canonical_name, value, "known")`
3. Use the RETURNED state_json for subsequent calls (it's updated)

### Rule 5: Categorical Value Mapping
Map user language to exact model values:
- "worn out", "worn", "bad" → "Worn Out"
- "new", "brand new", "fresh" → "New"  
- "good", "fine", "okay", "decent" → "Good"
- "weak", "dying", "dead", "drained" → "Weak" (Battery only)
- "good", "regular", "maintained" → "Good" (Maintenance_History)
- "average", "mixed" → "Average" (Maintenance_History only)
- "poor", "bad", "neglected" → "Poor" (Maintenance_History only)

### Rule 6: Numerical Feature Parsing
- Vehicle_Age: Extract years (e.g. "6 years old" → 6, "2 year old car" → 2)
- Reported_Issues: Count of problems (e.g. "has 3 issues" → 3, "several problems" → do NOT guess)
- Service_History: Number of service visits
- Accident_History: Number of accidents

## WORKFLOW

1. Call `get_current_state_summary(state_json)` to see what is already known
2. Call `get_model_features()` if you need to check available features
3. For each feature mentioned by the user:
   a. Call `validate_feature(feature_name)` to confirm it's in the schema
   b. If valid: call `update_feature_state(state_json, canonical, value, "known")`
   c. Use updated state_json for next tool call
4. For each feature the user said they don't know:
   a. Call `validate_feature(feature_name)` to confirm it's in the schema
   b. If valid: call `mark_feature_unavailable(state_json, canonical)`
5. Call `get_next_feature_to_ask(updated_state_json)` to find next question
6. Return a JSON response with the final state and extraction summary

## OUTPUT FORMAT
Return ONLY valid JSON:
{
  "state_json": "<updated AgentState JSON>",
  "extracted_features": {"Feature_Name": value, ...},
  "unavailable_features": ["Feature_Name", ...],
  "rejected_features": ["any_hallucinated_feature_name", ...],
  "next_feature": "<feature_name or null>",
  "next_feature_display": "<human name or null>",
  "next_feature_type": "<categorical|numerical|null>",
  "next_feature_valid_values": [<list or null>],
  "summary": "<brief summary of what was extracted>"
}
"""


def build_information_agent() -> LlmAgent:
    """Build and return the InformationAgent ADK LlmAgent."""
    return LlmAgent(
        name="InformationAgent",
        description=(
            "Extracts vehicle feature information from user messages, validates features "
            "against the ML model schema, updates the structured state, and determines "
            "the next feature to ask about. Never invents features outside the schema."
        ),
        model=get_fast_model(),
        instruction=INFORMATION_AGENT_INSTRUCTION,
        tools=[
            FunctionTool(get_model_features),
            FunctionTool(validate_feature),
            FunctionTool(update_feature_state),
            FunctionTool(mark_feature_unavailable),
            FunctionTool(get_next_feature_to_ask),
            FunctionTool(get_current_state_summary),
        ],
        output_key="information_result",
    )
