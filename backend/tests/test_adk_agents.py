"""
Automated Hallucination Prevention & Google ADK Multi-Agent Test Suite.

Tests cover:
1. Agent Initialization & Tooling
2. Test A: Vague symptom ("weird noise") -> Tire_Condition remains unknown, no hallucinated value
3. Test B: Single feature ("5 years old") -> Vehicle_Age=5, no prediction (< 2 features)
4. Test C: Second feature ("3 reported issues") -> Reported_Issues=3, question advances
5. Test D: Contextual short reply ("3" when asking Reported_Issues) -> sets Reported_Issues=3, clears current question
6. Test E: Two valid features -> Random Forest executes, risk comes strictly from ML
7. Test F: Third valid feature -> ML prediction recalculates with new prediction_id
8. Test G: Unsupported input ("Mileage is 30000") -> rejected, not added to state
9. Test H: Unrelated input ("bought a new helmet") -> no feature created
10. Test I: "I don't know" -> marked unavailable, never asked again
11. Test J: Rolling context & compaction -> confirmed state preserved
12. Test K: Judge Failure / Timeout -> unsafe response blocked, deterministic fallback returned
13. Test L: Repeated answer protection -> no question looping
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.agent.state.state import (
    AgentState,
    MIN_FEATURES_FOR_PREDICTION,
    CHECKPOINT_INPUT_RECEIVED,
    CHECKPOINT_FEATURES_EXTRACTED,
    CHECKPOINT_FEATURES_VALIDATED,
    CHECKPOINT_PREDICTION_GENERATED,
    CHECKPOINT_RESPONSE_GENERATED,
    CHECKPOINT_JUDGE_PASSED,
)
from backend.app.agent.state.feature_selector import (
    get_askable_features,
    get_next_question_feature,
    get_question_specification,
)
from backend.app.agent.tools.feature_tools import (
    validate_feature_name,
    validate_feature_value,
    apply_structured_extraction,
    extract_explicit_facts,
)
from backend.app.agent.tools.prediction_tools import execute_prediction_gate
from backend.app.agent.adk_agents.information_agent import build_information_agent
from backend.app.agent.adk_agents.response_agent import build_response_agent
from backend.app.agent.adk_agents.judge_agent import (
    build_judge_agent,
    run_deterministic_guards,
    generate_deterministic_fallback_response,
)
from backend.app.agent.adk_agents.controller_agent import build_controller_agent
from backend.app.agent.domain_guard import DomainGuard
from backend.app.ml.preprocessing import (
    SELECTED_FEATURES,
    FEATURE_PRIORITY,
    FEATURE_DISPLAY_NAMES,
    CATEGORICAL_VALUES,
    MaintenancePreprocessor,
)
from backend.app.context.compactor import ContextCompactor
from backend.app.context.manager import ContextManager
from backend.app.agent.feature_ranking import FeatureRankingEngine


# =============================================================================
# 1. ADK Agent Initialization
# =============================================================================

class TestADKAgentInitialization:
    """Verify all ADK agent builders instantiate valid LlmAgent instances."""

    def test_controller_agent_structure(self):
        controller = build_controller_agent()
        assert controller.name == "ControllerAgent"
        assert len(controller.tools) >= 3  # InputAgent, ResponseAgent, JudgeAgent

    def test_information_agent_structure(self):
        information_agent = build_information_agent()
        assert information_agent.name == "InformationAgent"
        assert information_agent.output_key == "information_result"

    def test_response_agent_structure(self):
        response_agent = build_response_agent()
        assert response_agent.name == "ResponseAgent"
        assert response_agent.output_key == "response_result"

    def test_judge_agent_structure(self):
        judge_agent = build_judge_agent()
        assert judge_agent.name == "JudgeAgent"
        assert judge_agent.output_key == "judge_result"


# =============================================================================
# 2. Hallucination Test Matrix (Tests A through L)
# =============================================================================

class TestHallucinationMatrix:
    """Exact test cases A through L required by specification."""

    def test_A_vague_symptom_no_invented_feature(self):
        """
        Test A: Input 'My tire has a weird noise.'
        Expected: Tire_Condition remains unknown, no invented tire condition, no prediction.
        """
        state = AgentState()
        # Vague statement is extracted as raw observation, NOT as Tire_Condition
        extraction = {
            "extracted_features": [],
            "unavailable_features": [],
            "raw_observations": ["tire has a weird noise"],
        }
        res = apply_structured_extraction(state, extraction)

        # Tire_Condition must not be confirmed
        assert state.feature_state["Tire_Condition"]["status"] == "unknown"
        assert state.feature_state["Tire_Condition"]["value"] is None
        assert "tire has a weird noise" in state.raw_observations

        # ML prediction must NOT run (< 2 features)
        pred_res = execute_prediction_gate(state)
        assert pred_res["allowed"] is False
        assert pred_res["prediction"] is None

    def test_B_single_feature_no_prediction(self):
        """
        Test B: Input 'My bike is 5 years old.'
        Expected: Vehicle_Age = 5, exactly one next valid question, no prediction (< 2 features).
        """
        state = AgentState()
        extraction = {
            "extracted_features": [{"name": "Vehicle_Age", "value": 5}],
            "unavailable_features": [],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert state.get_known_features() == {"Vehicle_Age": 5}
        assert state.known_feature_count() == 1

        # Hard programmatic gate blocks prediction
        pred_res = execute_prediction_gate(state)
        assert pred_res["allowed"] is False
        assert pred_res["prediction"] is None
        assert state.has_enough_for_prediction() is False

        # Next question should be chosen from remaining askable features
        next_q = get_next_question_feature(state)
        assert next_q is not None
        assert next_q != "Vehicle_Age"
        assert next_q in SELECTED_FEATURES

    def test_C_second_feature_advances_question(self):
        """
        Test C: Input 'I have 3 reported issues.'
        Expected: Reported_Issues = 3, question advances, Reported_Issues is not asked again.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")

        extraction = {
            "extracted_features": [{"name": "Reported_Issues", "value": 3}],
            "unavailable_features": [],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert state.get_known_features() == {"Vehicle_Age": 5, "Reported_Issues": 3}
        assert state.known_feature_count() == 2

        # Reported_Issues must not be in askable list
        askable = get_askable_features(state)
        assert "Reported_Issues" not in askable
        assert "Vehicle_Age" not in askable

        next_q = get_next_question_feature(state)
        assert next_q not in ["Vehicle_Age", "Reported_Issues"]

    def test_D_contextual_short_reply_resolves_current_question(self):
        """
        Test D: Input '3' when current question is Reported_Issues.
        Expected: Reported_Issues = 3, state updated, current question cleared, next question advances.
        """
        state = AgentState()
        # Assistant previously asked Reported_Issues
        state.mark_asked("Reported_Issues")
        assert state.current_question_feature == "Reported_Issues"

        # User replies simply "3" -> InputAgent resolves against current_question_feature
        extraction = {
            "extracted_features": [{"name": state.current_question_feature, "value": 3}],
            "unavailable_features": [],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert state.get_known_features()["Reported_Issues"] == 3
        # current_question_feature must now be cleared
        assert state.current_question_feature is None

        # Next question must not repeat Reported_Issues
        next_q = get_next_question_feature(state)
        assert next_q != "Reported_Issues"

    def test_E_two_valid_features_executes_rf_model(self):
        """
        Test E: Two valid features.
        Expected: Random Forest executes, prediction produced, risk score from ML.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")
        state.update_feature("Reported_Issues", 3, "known")

        assert state.known_feature_count() == 2
        assert state.has_enough_for_prediction() is True

        pred_res = execute_prediction_gate(state)
        assert pred_res["allowed"] is True
        assert pred_res["prediction"] is not None

        pred = pred_res["prediction"]
        assert pred["prediction"] in ["maintenance_required", "no_maintenance_needed"]
        assert 0.0 <= pred["model_score"] <= 1.0
        assert "prediction_id" in pred
        assert len(pred["prediction_id"]) > 10

    def test_F_third_feature_recalculates_prediction(self):
        """
        Test F: Third valid feature added.
        Expected: Prediction recalculated, unique prediction_id generated each time.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")
        state.update_feature("Reported_Issues", 3, "known")

        pred1 = execute_prediction_gate(state)["prediction"]
        assert pred1 is not None
        id1 = pred1["prediction_id"]

        # Add 3rd feature
        state.update_feature("Brake_Condition", "Worn Out", "known")
        assert state.known_feature_count() == 3

        pred2 = execute_prediction_gate(state)["prediction"]
        assert pred2 is not None
        id2 = pred2["prediction_id"]

        # Must have unique prediction IDs and more features used
        assert id1 != id2
        assert len(pred2["features_used"]) >= len(pred1["features_used"])

    def test_G_unsupported_feature_rejected(self):
        """
        Test G: Unsupported input 'My engine temperature is 30000.'
        Expected: Engine_Temperature is NOT added to state, no such question.
        """
        state = AgentState()

        is_valid, _ = validate_feature_name("Engine_Temperature")
        assert is_valid is False

        extraction = {
            "extracted_features": [{"name": "Engine_Temperature", "value": 30000}],
            "unavailable_features": [],
            "raw_observations": ["engine temperature is 30000"],
        }
        res = apply_structured_extraction(state, extraction)

        assert "Engine_Temperature" in res["rejected_features"]
        assert len(state.get_known_features()) == 0
        assert "Engine_Temperature" not in state.feature_state

    def test_H_unrelated_input_creates_no_feature(self):
        """
        Test H: Unrelated input 'I bought a new helmet yesterday.'
        Expected: No model feature created.
        """
        state = AgentState()
        extraction = {
            "extracted_features": [{"name": "Helmet_Condition", "value": "New"}],
            "unavailable_features": [],
            "raw_observations": ["bought a new helmet yesterday"],
        }
        res = apply_structured_extraction(state, extraction)

        assert "Helmet_Condition" in res["rejected_features"]
        assert state.known_feature_count() == 0

    def test_I_dont_know_marked_unavailable_never_reasked(self):
        """
        Test I: User says 'I don't know' for Battery_Status.
        Expected: Battery_Status becomes unavailable, never asked again.
        """
        state = AgentState()
        state.mark_asked("Battery_Status")

        extraction = {
            "extracted_features": [],
            "unavailable_features": ["Battery_Status"],
            "raw_observations": [],
        }
        apply_structured_extraction(state, extraction)

        assert "Battery_Status" not in state.declined_features
        assert state.feature_state["Battery_Status"]["status"] == "unavailable"
        assert "Battery_Status" in state.raw_user_inputs.get("unavailable_features", [])

        # Must never be in askable list
        askable = get_askable_features(state)
        assert "Battery_Status" not in askable

        next_q = get_next_question_feature(state)
        assert next_q != "Battery_Status"

    @pytest.mark.asyncio
    async def test_J_rolling_context_compaction_preserves_state(self):
        """
        Test J: Long conversation triggers compaction; confirmed state survives.
        """
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Vehicle Identity: 5yo Sedan\nKnown Features: Vehicle_Age=5, Reported_Issues=3"
        mock_router.generate = AsyncMock(return_value=mock_response)

        compactor = ContextCompactor(router=mock_router)
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")
        state.update_feature("Reported_Issues", 3, "known")

        messages = [
            {"role": "user", "content": "My car is 5 years old."},
            {"role": "assistant", "content": "Got it. How many issues?"},
            {"role": "user", "content": "3 reported issues."},
        ]

        summary = await compactor.compact_messages(
            messages=messages,
            feature_state_snapshot=state.to_context_dict(),
        )
        assert len(summary) > 0
        assert mock_router.generate.called
        assert state.get_known_features() == {"Vehicle_Age": 5, "Reported_Issues": 3}

    def test_K_judge_failure_enforces_deterministic_fallback(self):
        """
        Test K: Unsafe response rejected by guard / judge -> safe deterministic fallback returned.
        """
        state = AgentState()
        state.update_feature("Vehicle_Age", 5, "known")  # only 1 feature

        # Proposed response hallucinates a calculated risk and non-schema feature
        bad_response = "Your vehicle has high risk score 88% and mileage is 50000."

        is_valid, violations = run_deterministic_guards(
            response_text=bad_response,
            state=state,
            prediction=None,
            expected_next_feature="Reported_Issues",
        )
        assert is_valid is False
        assert len(violations) > 0

        # Deterministic fallback response must be safe & contain no hallucinated numbers
        fallback = generate_deterministic_fallback_response(
            state=state,
            prediction=None,
            next_feature="Reported_Issues",
        )
        assert "50000" not in fallback
        assert "88%" not in fallback
        assert "reported issues" in fallback.lower() or "issues" in fallback.lower()
        assert "vehicle age" in fallback.lower() and "5" in fallback

    def test_L_repeated_answer_protection_no_loop(self):
        """
        Test L: User answers same question; question selector advances and never loops.
        """
        state = AgentState()
        # Ask the current plan's first feature.
        q1 = get_next_question_feature(state)
        assert q1 in SELECTED_FEATURES
        state.mark_asked(q1)

        # User answers
        state.update_feature(q1, 2, "known")

        # Next question MUST advance
        q2 = get_next_question_feature(state)
        assert q2 != q1
        assert q2 in SELECTED_FEATURES


# =============================================================================
# 3. Deterministic Feature & Schema Validation Tests
# =============================================================================

class TestDeterministicValidationRules:
    """Test schema and categorical validation functions."""

    def test_schema_feature_count_matches_dataset_predictors(self):
        assert len(SELECTED_FEATURES) == 19

    def test_all_categorical_values_valid(self):
        assert "Good" in CATEGORICAL_VALUES["Maintenance_History"]
        assert "Worn Out" in CATEGORICAL_VALUES["Brake_Condition"]
        assert "Weak" in CATEGORICAL_VALUES["Battery_Status"]
        assert "New" in CATEGORICAL_VALUES["Tire_Condition"]

    def test_numerical_validation_rejects_negative(self):
        is_valid, val = validate_feature_value("Vehicle_Age", -5)
        assert is_valid is False
        assert val is None

    def test_numerical_validation_parses_natural_strings(self):
        is_valid, val = validate_feature_value("Vehicle_Age", "5 years")
        assert is_valid is True
        assert val == 5

    def test_vague_categorical_symptom_is_not_mapped_to_a_condition(self):
        is_valid, val = validate_feature_value("Brake_Condition", "squeaking")
        assert is_valid is False
        assert val is None

    def test_min_features_constant_is_strictly_two(self):
        assert MIN_FEATURES_FOR_PREDICTION == 2

    def test_message_parser_accepts_only_explicit_values(self):
        assert extract_explicit_facts("My tire has a weird noise.")["extracted_features"] == []
        assert extract_explicit_facts("My mileage is 30000.")["extracted_features"] == [{"name": "Mileage", "value": 30000}]
        assert extract_explicit_facts("I bought a new helmet yesterday.")["extracted_features"] == []
        facts = extract_explicit_facts("My bike is 5 years old.")
        assert facts["extracted_features"] == [{"name": "Vehicle_Age", "value": 5}]

    def test_labelled_fact_does_not_overwrite_a_different_pending_numeric_question(self):
        facts = extract_explicit_facts("4 problems", "Vehicle_Age")
        assert facts["extracted_features"] == [{"name": "Reported_Issues", "value": 4}]

    def test_current_question_is_not_repeated_before_an_answer(self):
        state = AgentState()
        state.set_feature_subset(["Reported_Issues"], "issues")
        state.update_feature("Reported_Issues", 4, status="known")
        assert get_next_question_feature(state) is None



class TestDynamicFeatureRanking:
    def test_brake_and_battery_intents_produce_different_leading_questions(self):
        brake = AgentState()
        battery = AgentState()
        brake_plan = FeatureRankingEngine.select_feature_subset(brake, "brake")
        battery_plan = FeatureRankingEngine.select_feature_subset(battery, "battery")
        brake.set_feature_subset(brake_plan, "brake")
        battery.set_feature_subset(battery_plan, "battery")

        assert brake_plan[0] == "Brake_Condition"
        assert battery_plan[0] == "Battery_Status"
        assert brake_plan != battery_plan
        assert get_next_question_feature(brake) == "Brake_Condition"
        assert get_next_question_feature(battery) == "Battery_Status"

    def test_prediction_audits_the_current_feature_values(self):
        state = AgentState()
        state.update_feature("Reported_Issues", 4, "known")
        state.update_feature("Brake_Condition", "Good", "known")
        prediction = execute_prediction_gate(state)["prediction"]
        assert prediction["feature_values_used"]["Reported_Issues"] == 4
        assert prediction["feature_values_used"]["Brake_Condition"] == "Good"
        assert prediction["feature_values_used"]["Battery_Status"] is None

        state.update_feature("Battery_Status", "New", "known")
        updated_prediction = execute_prediction_gate(state)["prediction"]
        assert updated_prediction["feature_values_used"]["Battery_Status"] == "New"
        assert updated_prediction["prediction_id"] != prediction["prediction_id"]


# =============================================================================
# 4. Universal NLP Feature Coverage (All 19 Features)
# =============================================================================

class TestUniversalNLPFeatureCoverage:
    """Exhaustive natural-language extraction verification across all 19 model features."""

    def test_nlp_mileage_extraction(self):
        # Stated units preserved as observations without model conversion
        f1 = extract_explicit_facts("My car has run 45k miles")
        assert f1["unverified_unit_features"]["Mileage"]["value"] == 45000
        assert f1["unverified_unit_features"]["Mileage"]["unit"] == "miles"
        assert not any(f["name"] == "Mileage" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("mileage is 50,000 km")
        assert f2["unverified_unit_features"]["Mileage"]["value"] == 50000
        assert f2["unverified_unit_features"]["Mileage"]["unit"] == "km"
        assert not any(f["name"] == "Mileage" for f in f2["extracted_features"])

        # Unitless contextual reply answering Mileage question directly
        f3 = extract_explicit_facts("55,000", current_question_feature="Mileage")
        assert any(f["name"] == "Mileage" and f["value"] == 55000 for f in f3["extracted_features"])

        # Unitless statement in message
        f4 = extract_explicit_facts("My mileage is 30000")
        assert any(f["name"] == "Mileage" and f["value"] == 30000 for f in f4["extracted_features"])

    def test_nlp_odometer_reading_extraction(self):
        f1 = extract_explicit_facts("odometer shows 62,000")
        assert any(f["name"] == "Odometer_Reading" and f["value"] == 62000 for f in f1["extracted_features"])

        f2 = extract_explicit_facts("odo reading is 55k")
        assert any(f["name"] == "Odometer_Reading" and f["value"] == 55000 for f in f2["extracted_features"])

        f3 = extract_explicit_facts("80000", current_question_feature="Odometer_Reading")
        assert any(f["name"] == "Odometer_Reading" and f["value"] == 80000 for f in f3["extracted_features"])

    def test_nlp_vehicle_age_extraction(self):
        f1 = extract_explicit_facts("I bought the car in 2021")
        assert any(f["name"] == "Vehicle_Age" and f["value"] == 5 for f in f1["extracted_features"])

        f2 = extract_explicit_facts("2022 model")
        assert any(f["name"] == "Vehicle_Age" and f["value"] == 4 for f in f2["extracted_features"])

        f3 = extract_explicit_facts("car is 4 years old")
        assert any(f["name"] == "Vehicle_Age" and f["value"] == 4 for f in f3["extracted_features"])

        f4 = extract_explicit_facts("5", current_question_feature="Vehicle_Age")
        assert any(f["name"] == "Vehicle_Age" and f["value"] == 5 for f in f4["extracted_features"])

    def test_nlp_reported_issues_extraction(self):
        f1 = extract_explicit_facts("zero issues")
        assert any(f["name"] == "Reported_Issues" and f["value"] == 0 for f in f1["extracted_features"])

        f2 = extract_explicit_facts("has 3 problems")
        assert any(f["name"] == "Reported_Issues" and f["value"] == 3 for f in f2["extracted_features"])

        f3 = extract_explicit_facts("no complaints")
        assert any(f["name"] == "Reported_Issues" and f["value"] == 0 for f in f3["extracted_features"])

        f4 = extract_explicit_facts("2", current_question_feature="Reported_Issues")
        assert any(f["name"] == "Reported_Issues" and f["value"] == 2 for f in f4["extracted_features"])

    def test_nlp_engine_size_extraction(self):
        f1 = extract_explicit_facts("engine size is 1.5L turbo")
        assert any(f["name"] == "Engine_Size" and f["value"] == 1500 for f in f1["extracted_features"])

        f2 = extract_explicit_facts("1500 cc engine")
        assert any(f["name"] == "Engine_Size" and f["value"] == 1500 for f in f2["extracted_features"])

        f3 = extract_explicit_facts("2.0 litres", current_question_feature="Engine_Size")
        assert any(f["name"] == "Engine_Size" and f["value"] == 2000 for f in f3["extracted_features"])

    def test_nlp_insurance_premium_extraction(self):
        f1 = extract_explicit_facts("insurance premium is 1200")
        assert any(f["name"] == "Insurance_Premium" and f["value"] == 1200 for f in f1["extracted_features"])

        f2 = extract_explicit_facts("insurance is 15k")
        assert any(f["name"] == "Insurance_Premium" and f["value"] == 15000 for f in f2["extracted_features"])

        f3 = extract_explicit_facts("2500", current_question_feature="Insurance_Premium")
        assert any(f["name"] == "Insurance_Premium" and f["value"] == 2500 for f in f3["extracted_features"])

    def test_nlp_service_history_extraction(self):
        f1 = extract_explicit_facts("serviced 4 times")
        assert any(f["name"] == "Service_History" and f["value"] == 4 for f in f1["extracted_features"])

        f2 = extract_explicit_facts("went to dealer 3 times")
        assert any(f["name"] == "Service_History" and f["value"] == 3 for f in f2["extracted_features"])

        f3 = extract_explicit_facts("5", current_question_feature="Service_History")
        assert any(f["name"] == "Service_History" and f["value"] == 5 for f in f3["extracted_features"])

    def test_nlp_accident_history_extraction(self):
        f1 = extract_explicit_facts("no accidents")
        assert any(f["name"] == "Accident_History" and f["value"] == 0 for f in f1["extracted_features"])

        f2 = extract_explicit_facts("crashed once")
        assert any(f["name"] == "Accident_History" and f["value"] == 1 for f in f2["extracted_features"])

        f3 = extract_explicit_facts("had 2 accidents")
        assert any(f["name"] == "Accident_History" and f["value"] == 2 for f in f3["extracted_features"])

    def test_nlp_fuel_efficiency_extraction(self):
        # Explicit kmpl preserved as observation
        f1 = extract_explicit_facts("fuel efficiency is 15 kmpl")
        assert f1["unverified_unit_features"]["Fuel_Efficiency"]["value"] == 15.0
        assert f1["unverified_unit_features"]["Fuel_Efficiency"]["unit"] == "km/l"
        assert not any(f["name"] == "Fuel_Efficiency" for f in f1["extracted_features"])

        # Generic MPG flagged as ambiguous
        f2 = extract_explicit_facts("gives 18 mpg")
        assert len(f2["ambiguities"]) == 1
        assert f2["ambiguities"][0]["type"] == "generic_mpg_ambiguity"
        assert f2["unverified_unit_features"]["Fuel_Efficiency"]["value"] == 18.0
        assert not any(f["name"] == "Fuel_Efficiency" for f in f2["extracted_features"])

        # Explicit L/100km preserved as observation
        f3 = extract_explicit_facts("fuel economy is 7 L/100km")
        assert f3["unverified_unit_features"]["Fuel_Efficiency"]["value"] == 7.0
        assert f3["unverified_unit_features"]["Fuel_Efficiency"]["unit"] == "l/100km"
        assert not any(f["name"] == "Fuel_Efficiency" for f in f3["extracted_features"])

        # Unitless contextual reply answering Fuel_Efficiency question directly
        f4 = extract_explicit_facts("20", current_question_feature="Fuel_Efficiency")
        assert any(f["name"] == "Fuel_Efficiency" and f["value"] == 20.0 for f in f4["extracted_features"])

    def test_nlp_tire_condition_extraction(self):
        f1 = extract_explicit_facts("tires are brand new")
        assert any(f["name"] == "Tire_Condition" and f["value"] == "New" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("tires are worn out")
        assert any(f["name"] == "Tire_Condition" and f["value"] == "Worn Out" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("good tyres")
        assert any(f["name"] == "Tire_Condition" and f["value"] == "Good" for f in f3["extracted_features"])

    def test_nlp_brake_condition_extraction(self):
        f1 = extract_explicit_facts("brakes are new")
        assert any(f["name"] == "Brake_Condition" and f["value"] == "New" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("worn out brake pads")
        assert any(f["name"] == "Brake_Condition" and f["value"] == "Worn Out" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("breaks are good")
        assert any(f["name"] == "Brake_Condition" and f["value"] == "Good" for f in f3["extracted_features"])

    def test_nlp_battery_status_extraction(self):
        f1 = extract_explicit_facts("battery is brand new")
        assert any(f["name"] == "Battery_Status" and f["value"] == "New" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("weak battery")
        assert any(f["name"] == "Battery_Status" and f["value"] == "Weak" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("batteru is good")
        assert any(f["name"] == "Battery_Status" and f["value"] == "Good" for f in f3["extracted_features"])

    def test_nlp_maintenance_history_extraction(self):
        f1 = extract_explicit_facts("maintenance is good")
        assert any(f["name"] == "Maintenance_History" and f["value"] == "Good" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("maintainance is avrage")
        assert any(f["name"] == "Maintenance_History" and f["value"] == "Average" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("poor maintenance")
        assert any(f["name"] == "Maintenance_History" and f["value"] == "Poor" for f in f3["extracted_features"])

    def test_nlp_fuel_type_extraction(self):
        f1 = extract_explicit_facts("fuel type is petrol")
        assert any(f["name"] == "Fuel_Type" and f["value"] == "Petrol" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("runs on diesel")
        assert any(f["name"] == "Fuel_Type" and f["value"] == "Diesel" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("it is an electric vehicle")
        assert any(f["name"] == "Fuel_Type" and f["value"] == "Electric" for f in f3["extracted_features"])

    def test_nlp_transmission_type_extraction(self):
        f1 = extract_explicit_facts("automatic transmission")
        assert any(f["name"] == "Transmission_Type" and f["value"] == "Automatic" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("stick shift")
        assert any(f["name"] == "Transmission_Type" and f["value"] == "Manual" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("manual gearbox")
        assert any(f["name"] == "Transmission_Type" and f["value"] == "Manual" for f in f3["extracted_features"])

    def test_nlp_owner_type_extraction(self):
        f1 = extract_explicit_facts("single owner")
        assert any(f["name"] == "Owner_Type" and f["value"] == "First" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("bought second hand")
        assert any(f["name"] == "Owner_Type" and f["value"] == "Second" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("3rd owner")
        assert any(f["name"] == "Owner_Type" and f["value"] == "Third" for f in f3["extracted_features"])

    def test_nlp_vehicle_model_extraction(self):
        f1 = extract_explicit_facts("vehicle model is suv")
        assert any(f["name"] == "Vehicle_Model" and f["value"] == "SUV" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("it's a sedan")
        assert any(f["name"] == "Vehicle_Model" and f["value"] == "Car" for f in f2["extracted_features"])

        f3 = extract_explicit_facts("it's a motorcycle")
        assert any(f["name"] == "Vehicle_Model" and f["value"] == "Motorcycle" for f in f3["extracted_features"])

        f4 = extract_explicit_facts("pickup truck")
        assert any(f["name"] == "Vehicle_Model" and f["value"] == "Truck" for f in f4["extracted_features"])

    def test_nlp_last_service_date_extraction(self):
        f1 = extract_explicit_facts("last serviced on 27 August 2025")
        assert any(f["name"] == "Last_Service_Date" and f["value"] == "2025-08-27" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("serviced on 15/05/2025")
        assert any(f["name"] == "Last_Service_Date" and f["value"] == "2025-05-15" for f in f2["extracted_features"])

    def test_nlp_warranty_expiry_date_extraction(self):
        f1 = extract_explicit_facts("warranty expiry date is 2027-12-31")
        assert any(f["name"] == "Warranty_Expiry_Date" and f["value"] == "2027-12-31" for f in f1["extracted_features"])

        f2 = extract_explicit_facts("warranty until 10 October 2026")
        assert any(f["name"] == "Warranty_Expiry_Date" and f["value"] == "2026-10-10" for f in f2["extracted_features"])


# =============================================================================
# 5. Universal NLP Semantic Boundaries & Safety Rules
# =============================================================================

class TestUniversalNLPSemanticBoundaries:
    """Verification of strict boundaries: symptoms vs condition, replacement vs condition, unsupported categories, contradictions, and ambiguities."""

    def test_symptom_isolation(self):
        state = AgentState()
        # Brakes symptom
        ext_brakes = extract_explicit_facts("My brakes are squeaking loudly and pedal feels soft")
        apply_structured_extraction(state, ext_brakes)
        assert "brakes" in state.symptoms
        assert state.feature_state["Brake_Condition"]["status"] == "unknown"

        # Battery symptom
        ext_batt = extract_explicit_facts("Battery keeps dying and engine cranks slowly")
        apply_structured_extraction(state, ext_batt)
        assert "battery" in state.symptoms
        assert state.feature_state["Battery_Status"]["status"] == "unknown"

        # Tire symptom
        ext_tires = extract_explicit_facts("Low tire pressure and vibration at high speed")
        apply_structured_extraction(state, ext_tires)
        assert "tires" in state.symptoms
        assert state.feature_state["Tire_Condition"]["status"] == "unknown"

    def test_replacement_history_isolation(self):
        state = AgentState()
        # Brake replacement
        ext_brakes = extract_explicit_facts("I replaced my brake pads 6 months ago")
        apply_structured_extraction(state, ext_brakes)
        assert "brakes" in state.raw_user_inputs.get("replacement_history", {})
        assert state.feature_state["Brake_Condition"]["status"] == "unknown"

        # Tire replacement
        ext_tires = extract_explicit_facts("I replaced my tyres one year ago")
        apply_structured_extraction(state, ext_tires)
        assert "tires" in state.raw_user_inputs.get("replacement_history", {})
        assert state.feature_state["Tire_Condition"]["status"] == "unknown"

        # Battery replacement
        ext_batt = extract_explicit_facts("Installed new battery last month")
        apply_structured_extraction(state, ext_batt)
        assert "battery" in state.raw_user_inputs.get("replacement_history", {})
        assert state.feature_state["Battery_Status"]["status"] == "unknown"

    def test_unsupported_categories_safeguard(self):
        state = AgentState()
        # 4th owner
        ext_owner = extract_explicit_facts("Vehicle is 4th owner")
        apply_structured_extraction(state, ext_owner)
        assert state.feature_state["Owner_Type"]["status"] == "unknown"
        assert any("4th owner" in obs for obs in state.raw_observations)

        # Hybrid
        ext_hybrid = extract_explicit_facts("The car is a hybrid")
        apply_structured_extraction(state, ext_hybrid)
        assert state.feature_state["Fuel_Type"]["status"] == "unknown"
        assert any("Hybrid" in obs for obs in state.raw_observations)

        # CNG
        ext_cng = extract_explicit_facts("Runs on CNG")
        apply_structured_extraction(state, ext_cng)
        assert state.feature_state["Fuel_Type"]["status"] == "unknown"
        assert any("CNG" in obs for obs in state.raw_observations)

        # Clean title
        ext_title = extract_explicit_facts("Vehicle has a clean title")
        apply_structured_extraction(state, ext_title)
        assert state.feature_state["Accident_History"]["status"] == "unknown"
        assert any("clean title" in obs for obs in state.raw_observations)

    def test_fuel_efficiency_vs_mileage_disambiguation(self):
        # 15 kmpl -> Fuel_Efficiency observation, not Mileage
        ext1 = extract_explicit_facts("My mileage is 15 kmpl")
        assert "Fuel_Efficiency" in ext1["unverified_unit_features"]
        assert "Mileage" not in ext1["unverified_unit_features"]
        assert len(ext1["extracted_features"]) == 0

        # 45,000 miles -> Mileage observation, not Fuel_Efficiency
        ext2 = extract_explicit_facts("Mileage is 45,000 miles")
        assert "Mileage" in ext2["unverified_unit_features"]
        assert "Fuel_Efficiency" not in ext2["unverified_unit_features"]
        assert len(ext2["extracted_features"]) == 0

        # Standalone 45000 -> ambiguity, neither
        ext3 = extract_explicit_facts("45000")
        assert len(ext3["ambiguities"]) == 1
        assert len(ext3["extracted_features"]) == 0

    def test_non_informative_distinctions_and_recovery(self):
        state = AgentState()
        state.mark_asked("Mileage")

        # "I don't know"
        ext_unk = extract_explicit_facts("I don't know", current_question_feature="Mileage")
        apply_structured_extraction(state, ext_unk)
        assert "Mileage" in state.raw_user_inputs.get("unknown_features", [])
        assert "Mileage" not in get_askable_features(state)

        # "I'll check and tell you" (pending)
        state.mark_asked("Last_Service_Date")
        ext_pend = extract_explicit_facts("I'll check and tell you later", current_question_feature="Last_Service_Date")
        apply_structured_extraction(state, ext_pend)
        assert "Last_Service_Date" in state.raw_user_inputs.get("pending_features", [])
        assert "Last_Service_Date" not in state.declined_features
        assert "Last_Service_Date" not in get_askable_features(state)

        # Late volunteer restores feature to known
        ext_vol = extract_explicit_facts("By the way, my mileage is 52000", current_question_feature=None)
        res_vol = apply_structured_extraction(state, ext_vol)
        assert "Mileage" in res_vol["updated_features"]
        assert state.feature_state["Mileage"]["status"] == "known"
        assert state.feature_state["Mileage"]["value"] == 52000
        assert "Mileage" not in state.declined_features

    def test_contradiction_detection_and_user_correction(self):
        state = AgentState()
        state.update_feature("Accident_History", 0, status="known")

        # Contradiction without correction wording -> rejected
        ext_contra = extract_explicit_facts("I had 2 accidents")
        res_contra = apply_structured_extraction(state, ext_contra)
        assert len(res_contra["contradictions"]) == 1
        assert state.feature_state["Accident_History"]["value"] == 0

        # User correction with explicit wording -> accepted
        ext_corr = extract_explicit_facts("Actually, I made a mistake, I had 2 accidents")
        res_corr = apply_structured_extraction(state, ext_corr)
        assert state.feature_state["Accident_History"]["value"] == 2
        assert len(res_corr["contradictions"]) == 0

    def test_ml_preprocessing_pipeline_invariance(self):
        state = AgentState()
        # Populate a mix of natural language facts
        ext1 = extract_explicit_facts("Vehicle is a 2021 model with 3 problems")
        apply_structured_extraction(state, ext1)

        ext2 = extract_explicit_facts("Brakes are good and fuel efficiency 15 kmpl")
        apply_structured_extraction(state, ext2)

        pp = MaintenancePreprocessor.from_config()
        df_transformed = pp.transform_feature_state(state.feature_state)

        # Must produce exactly 38 transformed columns
        assert df_transformed.shape == (1, 38)
        assert not df_transformed.isna().any().any()

    def test_vehicle_age_isolated_from_replacement_standalone(self):
        state = AgentState()
        ext = extract_explicit_facts("I replaced my tyres one year ago.")
        apply_structured_extraction(state, ext)
        assert not any(f["name"] == "Vehicle_Age" for f in ext["extracted_features"])
        assert state.feature_state["Vehicle_Age"]["status"] == "unknown"
        assert state.feature_state["Tire_Condition"]["status"] == "unknown"
        assert "tires" in state.raw_user_inputs.get("replacement_history", {})

    def test_vehicle_age_isolated_from_battery_replacement(self):
        state = AgentState()
        ext = extract_explicit_facts("My battery was replaced two years ago.")
        apply_structured_extraction(state, ext)
        assert not any(f["name"] == "Vehicle_Age" for f in ext["extracted_features"])
        assert state.feature_state["Vehicle_Age"]["status"] == "unknown"
        assert "battery" in state.raw_user_inputs.get("replacement_history", {})

    def test_vehicle_age_combined_sentence(self):
        state = AgentState()
        ext = extract_explicit_facts("My car is 5 years old, and I replaced my tyres one year ago.")
        apply_structured_extraction(state, ext)
        assert any(f["name"] == "Vehicle_Age" and f["value"] == 5 for f in ext["extracted_features"])
        assert state.feature_state["Vehicle_Age"]["value"] == 5
        assert state.feature_state["Vehicle_Age"]["status"] == "known"
        assert "tires" in state.raw_user_inputs.get("replacement_history", {})

    def test_vehicle_age_dynamic_year(self):
        from datetime import datetime
        current_year = datetime.now().year
        ext = extract_explicit_facts("Bought the car in 2021")
        assert any(f["name"] == "Vehicle_Age" and f["value"] == (current_year - 2021) for f in ext["extracted_features"])

    def test_contradiction_within_same_message(self):
        state = AgentState()
        ext = extract_explicit_facts("My brakes are good but completely worn out.")
        res = apply_structured_extraction(state, ext)
        assert len(ext["contradictions"]) == 1 or len(res["contradictions"]) == 1
        assert state.feature_state["Brake_Condition"]["status"] == "unknown"
        assert not any(f["name"] == "Brake_Condition" for f in ext["extracted_features"])

    def test_no_contradiction_with_negation(self):
        state = AgentState()
        ext = extract_explicit_facts("My brakes are good and not worn out.")
        res = apply_structured_extraction(state, ext)
        assert len(res["contradictions"]) == 0
        assert any(f["name"] == "Brake_Condition" and f["value"] == "Good" for f in ext["extracted_features"])
        assert state.feature_state["Brake_Condition"]["value"] == "Good"
        assert state.feature_state["Brake_Condition"]["status"] == "known"

    def test_negation_alone_does_not_set_feature(self):
        state = AgentState()
        ext = extract_explicit_facts("My battery is not weak.")
        apply_structured_extraction(state, ext)
        assert not any(f["name"] == "Battery_Status" for f in ext["extracted_features"])
        assert state.feature_state["Battery_Status"]["status"] == "unknown"

    def test_harmonious_conditions_no_contradiction(self):
        state = AgentState()
        ext = extract_explicit_facts("My tyres are new and working perfectly.")
        res = apply_structured_extraction(state, ext)
        assert len(res["contradictions"]) == 0
        assert any(f["name"] == "Tire_Condition" and f["value"] in ("New", "Good") for f in ext["extracted_features"])
        assert state.feature_state["Tire_Condition"]["status"] == "known"

    def test_four_uncertainty_states_distinction(self):
        # 1. Unknown
        s1 = AgentState()
        e1 = extract_explicit_facts("I don't know.", current_question_feature="Battery_Status")
        apply_structured_extraction(s1, e1)
        assert "Battery_Status" in s1.raw_user_inputs.get("unknown_features", [])
        assert "Battery_Status" not in s1.declined_features
        assert s1.feature_state["Battery_Status"]["status"] == "unknown"
        assert "Battery_Status" not in get_askable_features(s1)

        # 2. Unavailable
        s2 = AgentState()
        e2 = extract_explicit_facts("I can't remember.", current_question_feature="Battery_Status")
        apply_structured_extraction(s2, e2)
        assert "Battery_Status" in s2.raw_user_inputs.get("unavailable_features", [])
        assert "Battery_Status" not in s2.declined_features
        assert s2.feature_state["Battery_Status"]["status"] == "unavailable"
        assert "Battery_Status" not in get_askable_features(s2)

        # 3. Declined
        s3 = AgentState()
        e3 = extract_explicit_facts("I don't want to share.", current_question_feature="Battery_Status")
        apply_structured_extraction(s3, e3)
        assert "Battery_Status" in s3.declined_features
        assert "Battery_Status" not in s3.raw_user_inputs.get("unavailable_features", [])
        assert "Battery_Status" not in s3.raw_user_inputs.get("unknown_features", [])
        assert "Battery_Status" not in get_askable_features(s3)

        # 4. Pending
        s4 = AgentState()
        e4 = extract_explicit_facts("I'll check and tell you.", current_question_feature="Battery_Status")
        apply_structured_extraction(s4, e4)
        assert "Battery_Status" in s4.raw_user_inputs.get("pending_features", [])
        assert "Battery_Status" not in s4.declined_features
        assert "Battery_Status" not in get_askable_features(s4)

    def test_unresolved_state_cleared_upon_later_confirmation(self):
        state = AgentState()
        # Initial unknown
        ext_unk = extract_explicit_facts("I do not know", current_question_feature="Vehicle_Age")
        apply_structured_extraction(state, ext_unk)
        assert "Vehicle_Age" in state.raw_user_inputs.get("unknown_features", [])

        # Later confirmed
        ext_conf = extract_explicit_facts("It is 4 years old", current_question_feature=None)
        res_conf = apply_structured_extraction(state, ext_conf)
        assert state.feature_state["Vehicle_Age"]["status"] == "known"
        assert state.feature_state["Vehicle_Age"]["value"] == 4
        assert "Vehicle_Age" not in state.raw_user_inputs.get("unknown_features", [])

    def test_unverified_mileage_unit_preserved_as_observation(self):
        state = AgentState()
        ext = extract_explicit_facts("My car has run 45k miles.")
        apply_structured_extraction(state, ext)
        assert state.feature_state["Mileage"]["status"] == "unknown"
        assert state.raw_user_inputs.get("unverified_unit_features", {}).get("Mileage", {}).get("value") == 45000
        assert any("miles" in obs for obs in state.raw_observations)

    def test_generic_mpg_flagged_for_clarification(self):
        state = AgentState()
        ext = extract_explicit_facts("My car gives 18 mpg.")
        apply_structured_extraction(state, ext)
        assert len(ext["ambiguities"]) == 1
        assert ext["ambiguities"][0]["type"] == "generic_mpg_ambiguity"
        assert state.feature_state["Fuel_Efficiency"]["status"] == "unknown"

    def test_integer_feature_truncation_restored(self):
        from backend.app.agent.actions import AgentTools
        from unittest.mock import MagicMock
        tools = AgentTools(MagicMock())
        state = AgentState()
        res = tools.update_feature(state, "Vehicle_Age", 4.6, status="known")
        assert res["success"] is True
        assert state.feature_state["Vehicle_Age"]["value"] == 4

    def test_unavailable_and_declined_independent_decoupling_and_clearing(self):
        """
        Verify Item 1 decoupling requirements:
        - Unavailable does not enter declined_features.
        - Explicitly declined does not enter unavailable_features.
        - 'prefer not to share' and variations are captured as declined.
        - Unavailable features are not repeatedly requested.
        - Declined features are not repeatedly requested.
        - Both states clear correctly when a confirmed value arrives.
        - Existing unknown and pending behavior remains unchanged.
        """
        # A. Unavailable: "can't remember"
        st_unavail = AgentState()
        st_unavail.mark_asked("Vehicle_Age")
        ext_unavail = extract_explicit_facts("I can't recall", current_question_feature="Vehicle_Age")
        apply_structured_extraction(st_unavail, ext_unavail)
        assert st_unavail.feature_state["Vehicle_Age"]["status"] == "unavailable"
        assert "Vehicle_Age" in st_unavail.raw_user_inputs.get("unavailable_features", [])
        assert "Vehicle_Age" not in st_unavail.declined_features
        assert "Vehicle_Age" not in get_askable_features(st_unavail)

        # Confirm clears unavailable
        ext_conf_unavail = extract_explicit_facts("It's 6 years old")
        apply_structured_extraction(st_unavail, ext_conf_unavail)
        assert st_unavail.feature_state["Vehicle_Age"]["status"] == "known"
        assert st_unavail.feature_state["Vehicle_Age"]["value"] == 6
        assert "Vehicle_Age" not in st_unavail.raw_user_inputs.get("unavailable_features", [])

        # B. Declined: "prefer not to share"
        st_dec = AgentState()
        st_dec.mark_asked("Owner_Type")
        ext_dec = extract_explicit_facts("I prefer not to share that", current_question_feature="Owner_Type")
        apply_structured_extraction(st_dec, ext_dec)
        assert st_dec.feature_state["Owner_Type"]["status"] == "unavailable"
        assert "Owner_Type" in st_dec.declined_features
        assert "Owner_Type" not in st_dec.raw_user_inputs.get("unavailable_features", [])
        assert "Owner_Type" not in get_askable_features(st_dec)

        # Confirm clears declined
        ext_conf_dec = extract_explicit_facts("I am the first owner")
        apply_structured_extraction(st_dec, ext_conf_dec)
        assert st_dec.feature_state["Owner_Type"]["status"] == "known"
        assert st_dec.feature_state["Owner_Type"]["value"] == "First"
        assert "Owner_Type" not in st_dec.declined_features

        # C. Existing Unknown and Pending remain intact
        st_unk = AgentState()
        st_unk.mark_asked("Mileage")
        ext_unk = extract_explicit_facts("I don't know", current_question_feature="Mileage")
        apply_structured_extraction(st_unk, ext_unk)
        assert st_unk.feature_state["Mileage"]["status"] == "unknown"
        assert "Mileage" in st_unk.raw_user_inputs.get("unknown_features", [])
        assert "Mileage" not in st_unk.declined_features
        assert "Mileage" not in st_unk.raw_user_inputs.get("unavailable_features", [])

        st_pend = AgentState()
        st_pend.mark_asked("Last_Service_Date")
        ext_pend = extract_explicit_facts("I will check later", current_question_feature="Last_Service_Date")
        apply_structured_extraction(st_pend, ext_pend)
        assert st_pend.feature_state["Last_Service_Date"]["status"] == "unknown"
        assert "Last_Service_Date" in st_pend.raw_user_inputs.get("pending_features", [])
        assert "Last_Service_Date" not in st_pend.declined_features
        assert "Last_Service_Date" not in st_pend.raw_user_inputs.get("unavailable_features", [])


class TestContextualServiceTimingAndClarificationLifecycle:
    """
    Tests for contextual service timing handling, active question ambiguity
    lifecycle, attempt progression, and avoidance of premature unknown marking.
    """

    def test_first_timing_response_ambiguity_creation_and_observation(self):
        state = AgentState()
        state.update_feature("Vehicle_Age", 5)
        state.update_feature("Reported_Issues", 3)
        state.mark_asked("Maintenance_History")

        ext = extract_explicit_facts("it was done a year ago", current_question_feature="Maintenance_History")
        assert ext.get("service_recency") == "a year ago"
        assert not any(f.get("name") == "Maintenance_History" for f in ext.get("extracted_features", []))
        assert not any(f.get("name") == "Last_Service_Date" for f in ext.get("extracted_features", []))

        timing_amb = [a for a in ext.get("ambiguities", []) if a.get("type") == "active_question_timing_clarification"]
        assert len(timing_amb) == 1
        assert timing_amb[0]["feature"] == "Maintenance_History"
        assert timing_amb[0]["recency"] == "a year ago"

        res = apply_structured_extraction(state, ext)
        assert state.active_ambiguity is not None
        assert state.active_ambiguity["feature"] == "Maintenance_History"
        assert state.active_ambiguity["recency"] == "a year ago"
        assert state.active_ambiguity["attempt"] == 1
        assert state.raw_user_inputs.get("service_recency") == "a year ago"
        assert "Maintenance_History" not in state.raw_user_inputs.get("unknown_features", [])
        assert "Maintenance_History" not in state.declined_features
        assert state.feature_state["Maintenance_History"]["value"] is None

    def test_first_timing_response_deterministic_fallback_message(self):
        state = AgentState()
        state.update_feature("Vehicle_Age", 5)
        state.update_feature("Reported_Issues", 3)
        state.mark_asked("Maintenance_History")
        ext = extract_explicit_facts("it was done a year ago", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext)

        pred = {"prediction": "maintenance_required", "risk_level": "LOW", "model_score": 0.435}
        resp = generate_deterministic_fallback_response(
            state, pred, "Maintenance_History", active_ambiguity=state.active_ambiguity
        )
        assert "Thank you, I've noted that your vehicle was serviced a year ago." in resp
        assert "would you describe that maintenance history overall as Good, Average, or Poor?" in resp
        assert resp.count("?") == 1

    def test_repeated_timing_response_increments_attempt_and_clarifies(self):
        state = AgentState()
        state.update_feature("Vehicle_Age", 5)
        state.update_feature("Reported_Issues", 3)
        state.mark_asked("Maintenance_History")

        # Turn 1: "it was done a year ago"
        ext1 = extract_explicit_facts("it was done a year ago", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext1)
        assert state.active_ambiguity["attempt"] == 1

        # Turn 2: "it was done 1 year ago"
        ext2 = extract_explicit_facts("it was done 1 year ago", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext2)
        assert state.active_ambiguity["attempt"] == 2
        assert state.active_ambiguity["recency"] == "1 year ago"

        pred = {"prediction": "maintenance_required", "risk_level": "LOW", "model_score": 0.435}
        resp = generate_deterministic_fallback_response(
            state, pred, "Maintenance_History", active_ambiguity=state.active_ambiguity
        )
        assert "I understand your service was done 1 year ago." in resp
        assert "Service timing (when it was done) and maintenance quality (how well it has been maintained) are different details" in resp
        assert "Could you rate the overall maintenance history as Good, Average, or Poor? If you're not sure, feel free to say 'I don't know' or 'skip'." in resp
        assert resp.count("?") == 1

    def test_no_automatic_unknown_marking_on_repeated_timing(self):
        state = AgentState()
        state.mark_asked("Maintenance_History")

        for msg in ["it was done a year ago", "it was done 1 year ago", "it was serviced 12 months ago"]:
            ext = extract_explicit_facts(msg, current_question_feature="Maintenance_History")
            apply_structured_extraction(state, ext)

        assert state.active_ambiguity["attempt"] == 3
        assert "Maintenance_History" not in state.raw_user_inputs.get("unknown_features", [])
        assert "Maintenance_History" not in state.declined_features
        assert state.feature_state["Maintenance_History"]["value"] is None

    def test_explicit_unknown_marks_feature_unknown_and_clears_ambiguity(self):
        state = AgentState()
        state.mark_asked("Maintenance_History")
        ext1 = extract_explicit_facts("it was done a year ago", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext1)
        assert state.active_ambiguity is not None

        # User explicitly states unknown
        ext2 = extract_explicit_facts("I don't know", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext2)
        assert state.feature_state["Maintenance_History"]["status"] == "unknown"
        assert "Maintenance_History" in state.raw_user_inputs["unknown_features"]
        assert state.active_ambiguity is None

    def test_explicit_decline_marks_feature_declined_and_clears_ambiguity(self):
        state = AgentState()
        state.mark_asked("Maintenance_History")
        ext1 = extract_explicit_facts("it was done a year ago", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext1)
        assert state.active_ambiguity is not None

        # User explicitly declines
        ext2 = extract_explicit_facts("I prefer not to say", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext2)
        assert "Maintenance_History" in state.declined_features
        assert state.active_ambiguity is None

    def test_confirmed_categorical_clears_ambiguity_and_updates_feature(self):
        state = AgentState()
        state.mark_asked("Maintenance_History")
        ext1 = extract_explicit_facts("it was done a year ago", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext1)
        assert state.active_ambiguity is not None

        # User clarifies with categorical quality rating
        ext2 = extract_explicit_facts("Good", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext2)
        assert state.feature_state["Maintenance_History"]["status"] == "known"
        assert state.feature_state["Maintenance_History"]["value"] == "Good"
        assert state.active_ambiguity is None
        # Recency observation remains recorded
        assert state.raw_user_inputs.get("service_recency") == "a year ago"

    def test_relative_service_timing_preserves_recency_without_setting_last_service_date(self):
        cases = [
            "it was done a year ago",
            "it was serviced last year",
            "I had it serviced six months ago",
            "Maintenance was done recently",
        ]
        for c in cases:
            ext = extract_explicit_facts(c, current_question_feature="Maintenance_History")
            assert not any(f.get("name") == "Last_Service_Date" for f in ext.get("extracted_features", []))
            assert ext.get("service_recency") is not None

        # Exact calendar date DOES extract Last_Service_Date
        ext_exact = extract_explicit_facts("Last service was on 2026-03-12")
        assert any(f.get("name") == "Last_Service_Date" and f.get("value") == "2026-03-12" for f in ext_exact.get("extracted_features", []))

    def test_ambiguity_state_serialization_and_restoration(self):
        state = AgentState()
        state.active_ambiguity = {
            "type": "active_question_timing_clarification",
            "feature": "Maintenance_History",
            "recency": "a year ago",
            "attempt": 2,
        }
        json_data = state.to_json()
        restored = AgentState.from_json(json_data)
        assert restored.active_ambiguity == state.active_ambiguity
        assert restored.active_ambiguity["attempt"] == 2

    def test_ambiguity_does_not_leak_into_unrelated_turns(self):
        state = AgentState()
        state.mark_asked("Maintenance_History")
        ext1 = extract_explicit_facts("it was done a year ago", current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext1)
        assert state.active_ambiguity is not None

        # Next turn user provides fuel type instead of addressing maintenance history
        ext2 = extract_explicit_facts("The fuel type is petrol")
        apply_structured_extraction(state, ext2)
        assert state.active_ambiguity is None
        assert state.feature_state["Fuel_Type"]["status"] == "known"
        assert state.feature_state["Fuel_Type"]["value"] == "Petrol"
