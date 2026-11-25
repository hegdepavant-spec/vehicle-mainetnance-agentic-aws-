# 🚗 Vehicle Maintenance AI — Google ADK Multi-Agent System

A production-grade, full-stack **Vehicle Maintenance Multi-Agent System** powered by **Google Agent Development Kit (ADK)**. Combines genuine ADK multi-agent orchestration, pure ML maintenance prediction authority, dynamic feature questioning, context compaction, LLM-as-a-Judge evaluation, multi-LLM router failover, PostgreSQL/SQLite persistence, Redis caching, Google OAuth 2.0 emergency dispatch, and a ChatGPT-style React frontend.

> **CRITICAL ARCHITECTURAL GUARANTEE**: The LLM is **NEVER** the prediction authority. The trained ML model is the sole authority for maintenance predictions. The ADK `PredictionAgent` invokes the ML engine via an ADK `FunctionTool`.

---

## 🏗️ Multi-Agent Architecture (Google ADK)

The orchestration layer is built using real **Google Agent Development Kit (ADK)** primitives (`LlmAgent`, `FunctionTool`, `AgentTool`, `Runner`, `InMemorySessionService`):

```text
                           USER MESSAGE (FastAPI /api/chat)
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │           ADKAgentRunner            │
                       │     (Context Compaction Bridge)     │
                       └──────────────────┬──────────────────┘
                                          │ ADK Runner
                                          ▼
                       ┌─────────────────────────────────────┐
                       │           ControllerAgent           │
                       │     (Root ADK Orchestrator)         │
                       └─────────┬────────┬────────┬─────────┘
                                 │        │        │
            ┌────────────────────┘        │        └────────────────────┐
            ▼                             ▼                             ▼
 ┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
 │  InformationAgent   │       │   PredictionAgent   │       │    ResponseAgent    │
 │ (ADK Feature Extr.) │       │  (ADK ML Delegation)│       │(ADK Service Advisor)│
 └──────────┬──────────┘       └──────────┬──────────┘       └──────────┬──────────┘
            │                             │                             │
            ▼                             ▼                             ▼
    [Schema Validation]           [ML Engine Authority]         [Final Conversational]
   [State & Unavailable]         [Single Source of Truth]        [Natural Response]
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │     JudgeAgent      │
                               │(Post-Turn Evaluator)│
                               └─────────────────────┘
                                          │
                                          ▼
                            FINAL USER-FACING RESPONSE
```

---

## 🤖 Google ADK Agent Roles & Responsibilities

### 1. Controller Agent (`backend/app/agent/adk_agents/controller_agent.py`)
- **Type**: Root `LlmAgent`.
- **Orchestration**: Contains `InformationAgent`, `PredictionAgent`, and `ResponseAgent` registered via ADK `AgentTool`.
- **Responsibilities**:
  - Performs initial closed-domain verification via `check_domain`.
  - Coordinates feature extraction via `InformationAgent`.
  - Determines when ML prediction should occur (new info, explicit request, or all features exhausted).
  - Coordinates conversational response generation via `ResponseAgent`.
  - Never computes predictions or invents features.

### 2. Information / Feature Agent (`backend/app/agent/adk_agents/information_agent.py`)
- **Type**: Specialist `LlmAgent`.
- **ADK Tools**: `get_model_features`, `validate_feature`, `update_feature_state`, `mark_feature_unavailable`, `get_next_feature_to_ask`, `get_current_state_summary`.
- **Responsibilities**:
  - Dynamically inspects `models/feature_schema.json` as the authoritative source of truth.
  - Adheres to: *"Understand first. Extract second. Validate third. Update state fourth."*
  - **Global Extraction**: The active question never restricts extraction; all user-provided facts are captured.
  - **Coreference Resolution**: Resolves pronouns (*"it"*, *"that"*) across conversational history or active question context.
  - **Relative Temporal Representation**: Captures candidate intervals (*"one year ago"*, *"six months earlier"*, *"around Diwali"*) in `temporal_candidates` without fabricating calendar dates.
  - **Symptom & History Routing**: Routes sensory symptoms to `symptoms` and past actions to `replacement_history` without fabricating categorical wear conditions or altering vehicle age.
  - **Target-Specific Uncertainty**: Protects active questions when uncertainty is directed at specific components (e.g. *"I don't know when the battery was replaced"*).
  - Normalizes automotive typos using RapidFuzz with graceful Python standard library `difflib` fallback.
  - Rejects unsupported attributes (e.g. 4th owner, hybrid, CNG, clean title) into raw observations without forcing into model schema.

### 3. ML Prediction Agent (`backend/app/agent/adk_agents/prediction_agent.py`)
- **Type**: Specialist `LlmAgent`.
- **ADK Tools**: `run_ml_prediction`.
- **Responsibilities**:
  - Solely invokes the authoritative ML prediction engine (`backend/app/ml/predict.py`).
  - Generates a unique `prediction_id` per turn.
  - Returns probability, risk level (`HIGH` / `LOW`), features used/missing, and confidence metadata.
  - The LLM never computes or alters model scores.

### 4. Response Agent (`backend/app/agent/adk_agents/response_agent.py`)
- **Type**: Specialist `LlmAgent`.
- **ADK Tools**: `get_current_state_summary`, `get_next_feature_to_ask`, `get_model_features`.
- **Responsibilities**:
  - Formulates natural, conversational responses acting as AutoCare AI service advisor.
  - Naturally phrases single feature questions derived dynamically from `models/feature_schema.json`.
  - Explains predictions without revealing raw numeric model scores (handled by UI cards).
  - Never asks about unavailable or already known features.

### 5. Judge Agent (`backend/app/agent/adk_agents/judge_agent.py`)
- **Type**: Independent Evaluator `LlmAgent`.
- **ADK Tools**: `get_model_features`.
- **Responsibilities**:
  - Evaluates the final response and agent execution trace post-turn.
  - Checks for hallucinations, invalid feature questions, repeated unavailable questions, and ML consistency.
  - Emits structured evaluation JSON (`score: 1-5`, `passed: bool`, `hallucination: bool`, `reason: str`).
  - Operates as a read-only auditor without modifying conversation or user state.

---

## 📊 Dataset & ML Pipeline

### Dataset Overview
- **File**: `data/vehicle_maintenance_data.csv`
- **Size**: 50,000 rows, 20 columns
- **Target**: `Need_Maintenance` (Binary: 0 = No, 1 = Yes)

### Exploratory Data Analysis & Feature Selection
1. **Authoritative 19-Feature Schema (`models/feature_schema.json`)**:
   - **Categorical (8)**: `Brake_Condition`, `Battery_Status`, `Tire_Condition`, `Maintenance_History`, `Fuel_Type`, `Transmission_Type`, `Owner_Type`, `Vehicle_Model`.
   - **Numerical (9)**: `Reported_Issues`, `Vehicle_Age`, `Mileage`, `Odometer_Reading`, `Engine_Size`, `Fuel_Efficiency`, `Insurance_Premium`, `Service_History`, `Accident_History`.
   - **Date (2)**: `Last_Service_Date`, `Warranty_Expiry_Date`.

2. **Preprocessing Pipeline (`MaintenancePreprocessor`)**:
   - Transforms partial inputs into an exact **38-column model matrix** ($19 \text{ features} + 19 \text{ missingness indicators}$).
   - Categoricals mapped to ordinal integers (`0.0`, `1.0`, `2.0`), with missing/unknown imputed to `-1.0` and `Feature_missing = 1`.
   - Dates converted to ordinal timestamps; numerical features imputed to `-1.0` if unknown.
   - Programmatic Prediction Gate executes inference as soon as $\ge 2$ features are confirmed.

### Model Performance & Production Artifact
- **Production Classifier**: **Random Forest** (`models/maintenance_model.joblib`, version 2.0.0)
- **Accuracy**: 96.45% (Test Split) | **ROC-AUC**: 1.0000 | **Precision**: 100.00% | **F1 Score**: 0.9776
- **Calibration**: Brier Score: 0.0360, Expected Calibration Error (ECE): 0.1006
- **Explainability**: Integrated SHAP TreeExplainer delivering local feature contributions for risk factors.

---

## 🧠 Context Management & Compaction

- **Rolling Context Window**: Configurable recent message buffer (`max_recent_messages = 20`).
- **Context Compactor (`backend/app/context/compactor.py`)**:
  - Automatically activates when turn count exceeds threshold (`compaction_threshold = 30`).
  - Condenses older turns into a structured summary (vehicle identity, known features, unavailable features, predictions, key symptoms).
  - Enforces `STRUCTURED STATE > COMPACTED SUMMARY` precedence.

---

## 🔀 Multi-LLM Provider Router & ADK Model Bridge

- **`RouterBackedModel` (`backend/app/agent/adk_agents/model_bridge.py`)**: Custom ADK `BaseLlm` bridge that translates ADK `LlmRequest` to the project's multi-provider router.
- **Failover Chain**: Primary (Groq/OpenAI) → Secondary (Fallback) → Tertiary provider.
- **Circuit Breaker**: Exponential backoff on rate limits (HTTP 429), 5-minute circuit breaker after 3 failures.

---

## 🔐 Google OAuth 2.0 & Emergency Gmail Dispatch

- **Google OAuth 2.0 PKCE**: Secure Google Sign-In with server-side Fernet-encrypted token storage.
- **Emergency Engine (`backend/app/emergency/emergency_engine.py`)**:
  - Detects critical conditions (`Brake_Condition = 'Worn Out'`, `Battery_Status = 'Weak'`).
  - Renders an Emergency Alert modal in the frontend.
  - Dispatches emergency alert emails to saved emergency contacts via the user's Gmail API credentials.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm

### 1. Backend Setup

```bash
# Clone repository and enter project root
cd vehicle_maintainence_agentic

# Configure environment variables
cp .env.example .env
# Edit .env with your LLM API Key(s) (e.g., Groq / OpenAI) and Google OAuth credentials

# Install dependencies (from root or backend directory)
pip install -r backend/requirements.txt
# (or from root: pip install -r requirements.txt)

# Option A: From project root (Recommended)
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 7860

# Option B: From inside the backend directory
# (PowerShell)
$env:PYTHONPATH=".."
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 7860
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Comprehensive Test Suite

Run the full automated test suite (**158/158 tests passing**) covering Universal Conversational NLP handling, Google ADK agents, Hypothesis Grounding, OAuth PKCE, Token Encryption, Emergency Engine, Gmail Dispatch, ML prediction authority, and Security:

```bash
# Run all tests
python -m pytest backend/tests/ -v

# Run Universal Conversational NLP intelligence tests
python -m pytest backend/tests/test_universal_nlp_intelligence.py -v

# Run Google ADK Multi-Agent tests
python -m pytest backend/tests/test_adk_agents.py -v
```

### Test Suite Summary (169 Passed):
- **Universal Conversational NLP Suite (`test_universal_nlp_intelligence.py` - 32 tests)**:
  - Off-target relative service timing responses (*"Service was done two years ago"* under `Maintenance_History` does not guess `Poor`, captures recency & candidate window, keeps question open).
  - Relative temporal normalization to ISO dates (`YYYY-MM-DD`) with approximation indicators (`is_exact: False`, e.g. *"one year ago"*, *"around November last year"*).
  - Conversational service timing recognition (*"last year i did"*) under `Maintenance_History` with active ambiguity clarification.
  - Component replacement and timing capture (*"i just changed my battery 2 months ago"*) without asserting `"New"`.
  - Subjective/vague component condition handling (*"battery is kind of old"*) triggering clarification without mapping to `"Weak"`.
  - Component condition phrasing (*"brake is working properly"*, *"works fine"*, *"functioning properly"*) mapping cleanly to `"Good"`.
  - Approximate numerical answers (*"approximately 5"*) under `Service_History`, `Reported_Issues`, `Vehicle_Age` with approximation indicators.
  - Negative history parsing (*"no accidents"*) mapping to `Accident_History = 0`.
  - Component age safeguard (*"The battery is almost 3 years old."*) protected from setting `Vehicle_Age`.
  - Engine thermal symptom detection (*"The engine temperature rises during long drives."*) routed to `symptoms["engine"]`.
  - Unremembered service dates (*"I don't remember when I last serviced my car."*) marked `unavailable_features["Last_Service_Date"]`.
  - Natural language aliases (*"distance driven"*, *"number of services"*, *"number of accidents"*).
  - Typo tolerance in corrections across conversational turns (*"Actually, it was done one year agoo"*).
  - Multi-attribute extraction in a single sentence (*"battery is weak, brakes make grinding noise"*).
  - Component replacement vs vehicle age (*"tyres replaced 3 months ago"* preserves `Vehicle_Age` & `Tire_Condition`).
  - Target-specific uncertainty (*"I don't know when the battery was replaced"* protects active question).
  - Mixed sentiment with subtle symptoms (*"car has been fine, except engine struggles to start in the morning"*).
  - Specification Examples A through E (tyre replacement age, covered mileage, dying battery symptom, unknown oil change, 3-attribute multi-extraction).
- **Google ADK Multi-Agent Suite (`test_adk_agents.py` - 37 tests)**:
  - ADK agent structure verification (`ControllerAgent`, `InformationAgent`, `PredictionAgent`, `ResponseAgent`, `JudgeAgent`).
  - Dynamic adaptive questioning without hardcoded counts.
  - Multi-feature extraction and structured state updates.
  - Natural *"I don't know"* handling (`unavailable` status, never asked again).
  - Pure ML prediction authority via ADK tool.
  - Context compaction & rolling window verification.
  - Closed-domain guard validation.
- **Hypothesis Grounding & Date Normalization Suite (`test_hypothesis_grounding.py` - 19 tests)**:
  - Grounded symptom routing, natural service date normalization (`YYYY-MM-DD`), calendar date validation.
- **Security & OAuth Suite (`test_security.py`, `test_oauth.py`, `test_token_manager.py`)**:
  - OAuth 2.0 PKCE, Fernet encryption at rest, token refresh.
  - Prompt injection immunity, user isolation, emergency contact CRUD.
- **ML & Consistency Suite (`test_ml.py`, `test_prediction_consistency.py`)**:
  - Single ML prediction per turn, probability score consistency, 38-column preprocessor verification.
- **Emergency Flow Suite (`test_emergency_engine.py`, `test_gmail_service.py`, `test_emergency_flow.py`)**:
  - Explicit rule-based emergency triggering and Gmail dispatch.

---

## 📁 Repository Structure

```text
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── adk_agents/          # Google ADK Multi-Agent Implementation
│   │   │   │   ├── controller_agent.py # Root ControllerAgent (LlmAgent + AgentTools)
│   │   │   │   ├── information_agent.py# InformationAgent (Feature extraction & validation)
│   │   │   │   ├── prediction_agent.py # PredictionAgent (ML prediction delegation)
│   │   │   │   ├── response_agent.py   # ResponseAgent (Conversational service advisor)
│   │   │   │   ├── judge_agent.py      # JudgeAgent (Post-turn evaluator)
│   │   │   │   ├── model_bridge.py     # RouterBackedModel (ADK BaseLlm bridge)
│   │   │   │   ├── tools.py            # ADK FunctionTool functions
│   │   │   │   └── runner.py           # ADKAgentRunner (Session & turn orchestrator)
│   │   │   ├── state/
│   │   │   │   ├── state.py            # AgentState (State: known, temporal, symptoms, replacements)
│   │   │   │   └── feature_selector.py # Deterministic next-question selection policy
│   │   │   ├── tools/
│   │   │   │   ├── feature_tools.py    # Typo normalizer, coreference, intervals, & extraction
│   │   │   │   └── prediction_tools.py # Deterministic ML execution gate
│   │   │   ├── domain_guard.py         # Closed-domain boundary validator
│   │   │   ├── hypothesis_grounding.py # Symptom grounding & observation routing
│   │   │   └── question_guard.py       # Hard feature boundary validator
│   │   ├── api/                        # FastAPI REST & SSE Streaming Endpoints
│   │   ├── context/                    # ContextManager & ContextCompactor
│   │   ├── emergency/                  # Emergency Engine & Gmail Dispatch
│   │   ├── llm/                        # Multi-LLM Router & Providers
│   │   ├── ml/                         # ML Model Predictor & Preprocessing Pipeline
│   │   └── main.py                     # FastAPI Application Entry Point
│   ├── tests/                          # 169 Automated Tests
│   └── requirements.txt                # Python Backend Dependencies
├── frontend/                           # React + Vite + TypeScript Frontend
├── models/
│   ├── feature_schema.json             # Authoritative 19-Feature Definition
│   ├── maintenance_model.joblib        # Trained Random Forest Model (v2.0.0)
│   ├── preprocessing_pipeline.joblib   # Fitted 38-Column Preprocessor Pipeline
│   └── model_metadata.json             # Hyperparameters, Metrics & Priorities
├── data/                               # Dataset & Samples
├── requirements.txt                    # Root Requirements Pointer
└── README.md                           # Documentation
```
