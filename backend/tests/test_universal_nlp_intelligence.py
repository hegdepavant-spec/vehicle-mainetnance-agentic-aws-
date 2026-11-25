"""
Comprehensive test suite for Universal Conversational NLP Intelligence:
1. Understand first. Extract second. Validate third. Update state fourth.
2. The active question never restricts NLP extraction.
3. Distinguish relative service timing from maintenance quality without fabricating dates.
4. Correct antecedent resolution across turns.
5. Typo tolerance across automotive lexicon.
6. Target-specific uncertainty vs active question uncertainty.
7. Symptoms vs physical wear conditions.
8. Replacement history vs current condition vs vehicle age.
9. 19-feature schema preservation and question budget safety.
"""

import re
from datetime import datetime, date
import pytest
from dateutil.relativedelta import relativedelta

from backend.app.agent.state.state import AgentState
from backend.app.agent.tools.feature_tools import (
    extract_explicit_facts,
    constrain_extraction_to_explicit_facts,
    apply_structured_extraction,
    normalize_typos_and_informal_tokens,
    resolve_contextual_antecedents,
    parse_temporal_expressions_and_intervals,
    validate_feature_name,
)
from backend.app.ml.feature_schema import MODEL_FEATURE_SCHEMA


# ==============================================================================
# Mandatory Scenarios (Scenarios 1 to 6)
# ==============================================================================

class TestMandatoryScenarios:

    def test_scenario_1_off_target_service_timing(self):
        """
        AI asks: "How would you describe the vehicle's maintenance history — Good, Average, Poor?"
        User: "Service was done two years ago."
        Expectation:
        - Maintenance_History is NOT set to Poor/Average/Good.
        - Maintenance_History is NOT marked unknown or unavailable.
        - Last_Service_Date is NOT written as a confirmed exact calendar date in feature_state.
        - Candidate interval is recorded in temporal_candidates.
        - Entity observation is recorded in entity_observations["service_recency"].
        """
        state = AgentState(conversation_id="test_sc1")
        history = [
            {"role": "assistant", "content": "How would you describe your vehicle's maintenance history — Good, Average, or Poor?"}
        ]
        user_msg = "Service was done two years ago."

        extraction = extract_explicit_facts(
            user_msg,
            current_question_feature="Maintenance_History",
            conversation_history=history,
        )

        assert extraction["extracted_features"] == []
        assert "Maintenance_History" not in extraction["unknown_features"]
        assert "Maintenance_History" not in extraction["unavailable_features"]
        assert "Last_Service_Date" in extraction["temporal_candidates"]
        cand = extraction["temporal_candidates"]["Last_Service_Date"]
        assert cand["is_exact"] is False
        assert "2 years ago" in cand["approximate_date"] or "year" in cand["raw_text"]
        assert "service_recency" in extraction["entity_observations"]

        # Apply to state
        apply_structured_extraction(state, extraction, raw_user_message=user_msg)
        assert state.feature_state["Maintenance_History"]["status"] == "unknown"
        assert state.feature_state["Maintenance_History"]["value"] is None
        assert state.feature_state["Last_Service_Date"]["value"] is None
        assert "Last_Service_Date" in state.temporal_candidates
        assert "service_recency" in state.entity_observations

    def test_scenario_2_typo_in_correction_across_turns(self):
        """
        Turn 1: "Service was done two years ago"
        Turn 2: "Actually, it was done one year agoo"
        Expectation:
        - Typo 'agoo' is normalized.
        - is_correction is True.
        - service_recency observation is revised to 1 year ago without duplicated stale observations.
        """
        state = AgentState(conversation_id="test_sc2")
        
        # Turn 1
        t1_msg = "Service was done two years ago."
        ext1 = extract_explicit_facts(t1_msg, current_question_feature="Maintenance_History")
        apply_structured_extraction(state, ext1, raw_user_message=t1_msg)
        assert any(y in state.entity_observations["service_recency"] for y in ("two years", "2 years"))

        # Turn 2
        history = [
            {"role": "user", "content": t1_msg},
            {"role": "assistant", "content": "Understood, your last service was around 2 years ago. Could you also tell me if the maintenance history was Good, Average, or Poor?"}
        ]
        t2_msg = "Actually, it was done one year agoo"
        ext2 = extract_explicit_facts(
            t2_msg,
            current_question_feature="Maintenance_History",
            conversation_history=history,
        )
        assert ext2["is_correction"] is True
        assert "Last_Service_Date" in ext2["temporal_candidates"]
        
        apply_structured_extraction(state, ext2, raw_user_message=t2_msg)
        assert any(y in state.entity_observations["service_recency"] for y in ("one year", "1 year"))
        # Verify no duplicate entries in state
        assert isinstance(state.entity_observations["service_recency"], str)

    def test_scenario_3_multiple_facts_in_one_message(self):
        """
        User: "battery is weak, brakes make grinding noise"
        Expectation:
        - Battery_Status confirmed as "Weak".
        - Brake grinding noise captured as symptom in symptoms["brakes"].
        - Brake_Condition is NOT set to "Worn Out".
        """
        state = AgentState(conversation_id="test_sc3")
        user_msg = "battery is weak, brakes make grinding noise"
        ext = extract_explicit_facts(user_msg)

        extracted_dict = {f["name"]: f["value"] for f in ext["extracted_features"]}
        assert extracted_dict.get("Battery_Status") == "Weak"
        assert "Brake_Condition" not in extracted_dict
        assert "brakes" in ext["symptoms"]
        assert "grinding" in ext["symptoms"]["brakes"]

        apply_structured_extraction(state, ext, raw_user_message=user_msg)
        assert state.feature_state["Battery_Status"]["value"] == "Weak"
        assert state.feature_state["Battery_Status"]["status"] == "known"
        assert state.feature_state["Brake_Condition"]["status"] == "unknown"
        assert state.symptoms.get("brakes") is not None

    def test_scenario_4_component_replacement_vs_vehicle_age(self):
        """
        User: "tyres replaced 3 months ago"
        Expectation:
        - Tire replacement captured in replacement_history["tires"].
        - Vehicle_Age is NOT set (neither 3 nor 0.25).
        - Tire_Condition is NOT set to "New".
        """
        state = AgentState(conversation_id="test_sc4")
        user_msg = "tyres replaced 3 months ago"
        ext = extract_explicit_facts(user_msg)

        extracted_dict = {f["name"]: f["value"] for f in ext["extracted_features"]}
        assert "Vehicle_Age" not in extracted_dict
        assert "Tire_Condition" not in extracted_dict
        assert "tires" in ext["replacement_history"]

        apply_structured_extraction(state, ext, raw_user_message=user_msg)
        assert state.feature_state["Vehicle_Age"]["status"] == "unknown"
        assert state.feature_state["Tire_Condition"]["status"] == "unknown"
        assert "tires" in state.replacement_history

    def test_scenario_5_unavailable_historical_info(self):
        """
        Active question: Maintenance_History
        User: "I don't know when the battery was replaced"
        Expectation:
        - Maintenance_History is NOT marked unknown or unavailable.
        - Does NOT invent dates for Last_Service_Date.
        - Battery replacement uncertainty is recognized as specific/targeted.
        """
        state = AgentState(conversation_id="test_sc5")
        user_msg = "I don't know when the battery was replaced"
        ext = extract_explicit_facts(user_msg, current_question_feature="Maintenance_History")

        assert "Maintenance_History" not in ext["unknown_features"]
        assert "Maintenance_History" not in ext["unavailable_features"]
        assert ext["extracted_features"] == []
        assert "raw_observations" in ext

        apply_structured_extraction(state, ext, raw_user_message=user_msg)
        assert state.feature_state["Maintenance_History"]["status"] == "unknown"
        assert "Maintenance_History" not in state.raw_user_inputs.get("unknown_features", [])

    def test_scenario_6_mixed_sentiment_with_subtle_symptom(self):
        """
        User: "The car has been fine, except the engine struggles to start in the morning."
        Expectation:
        - Captures engine symptom in symptoms["engine"].
        - Does NOT corrupt Engine_Size or other features.
        """
        state = AgentState(conversation_id="test_sc6")
        user_msg = "The car has been fine, except the engine struggles to start in the morning."
        ext = extract_explicit_facts(user_msg)

        assert "engine" in ext["symptoms"]
        assert "struggles to start" in ext["symptoms"]["engine"]
        assert "Engine_Size" not in [f["name"] for f in ext["extracted_features"]]

        apply_structured_extraction(state, ext, raw_user_message=user_msg)
        assert state.symptoms.get("engine") is not None
        assert state.feature_state["Engine_Size"]["value"] is None


# ==============================================================================
# Generalization & Edge Cases Matrix
# ==============================================================================

class TestGeneralizationMatrix:

    def test_automotive_typos_normalization(self):
        """Common typos in automotive terms normalize cleanly."""
        assert "battery" in normalize_typos_and_informal_tokens("battry is low")
        assert "brakes" in normalize_typos_and_informal_tokens("braks squeaking")
        assert "serviced" in normalize_typos_and_informal_tokens("servisd last month")
        assert "1 year ago" in normalize_typos_and_informal_tokens("done 1 year agoo")

    def test_coreference_resolution_with_antecedent(self):
        """Resolves pronouns like 'it' to antecedent in conversation history."""
        history = [
            {"role": "user", "content": "I am worried about my battery."},
            {"role": "assistant", "content": "When was the battery last replaced?"}
        ]
        target = resolve_contextual_antecedents("it was replaced last month", history)
        assert target == "battery"

    def test_coreference_resolution_ambiguity(self):
        """When multiple components are in history, 'it' is ambiguous."""
        history = [
            {"role": "user", "content": "The battery and the brakes both feel weird."},
            {"role": "assistant", "content": "Tell me more about them."}
        ]
        target = resolve_contextual_antecedents("it was changed recently", history)
        assert target == "AMBIGUOUS"

    def test_informal_and_seasonal_temporal_parsing(self):
        """Parses informal intervals like 'around Diwali', 'since last year', 'every six months'."""
        diwali = parse_temporal_expressions_and_intervals("around Diwali")
        assert diwali is not None
        assert "Diwali" in diwali["raw"]
        assert diwali["is_exact"] is False

        since = parse_temporal_expressions_and_intervals("since last year")
        assert since is not None
        assert since["is_exact"] is False

        freq = parse_temporal_expressions_and_intervals("every six months")
        assert freq is not None
        assert "every 6 months" in freq["raw"] or "six months" in freq["raw"]

    def test_negation_handling(self):
        """Negations like 'brakes are not worn' should not trigger 'Worn Out'."""
        user_msg = "the brakes are not worn out"
        ext = extract_explicit_facts(user_msg)
        extracted_dict = {f["name"]: f["value"] for f in ext["extracted_features"]}
        assert extracted_dict.get("Brake_Condition") != "Worn Out"

    def test_subtle_intent_general_drivability(self):
        """Expressions like 'doesn't feel right when I drive' map to general symptoms."""
        user_msg = "The car doesn't feel right when I drive"
        ext = extract_explicit_facts(user_msg)
        assert "general" in ext["symptoms"]

    def test_unsupported_unmappable_facts(self):
        """Unmappable facts like 'I painted the roof black yesterday' are stored in raw_observations."""
        user_msg = "I painted the roof black yesterday"
        ext = extract_explicit_facts(user_msg)
        assert len(ext["extracted_features"]) == 0
        assert any("painted the roof" in obs for obs in ext["raw_observations"])


# ==============================================================================
# Specification Examples Verification (Examples A to E)
# ==============================================================================

class TestPromptNaturalLanguageExamples:

    def test_example_a_tyre_replacement_one_year_ago(self):
        """
        User: 'I replaced my tyres one year ago.'
        - Tyre replacement age: 1 year.
        - Physical tyre condition: UNKNOWN.
        - Does NOT assume 1-year-old tyre is in good condition.
        """
        state = AgentState(conversation_id="test_ex_a")
        user_msg = "I replaced my tyres one year ago."
        ext = extract_explicit_facts(user_msg)

        assert "tires" in ext["replacement_history"]
        assert "Tire_Condition" not in [f["name"] for f in ext["extracted_features"]]
        assert "Vehicle_Age" not in [f["name"] for f in ext["extracted_features"]]

        apply_structured_extraction(state, ext, raw_user_message=user_msg)
        assert state.feature_state["Tire_Condition"]["status"] == "unknown"
        assert state.feature_state["Tire_Condition"]["value"] is None
        assert "tires" in state.replacement_history

    def test_example_b_mileage_covered_distance(self):
        """
        User: 'My car has covered around 60 thousand kilometres.'
        - Captures cumulative distance: 60,000 km.
        """
        user_msg = "My car has covered around 60 thousand kilometres."
        ext = extract_explicit_facts(user_msg)
        assert ext["unverified_unit_features"]["Mileage"]["value"] == 60000
        assert ext["unverified_unit_features"]["Mileage"]["unit"] == "kilometres"

    def test_example_c_battery_dying_symptom(self):
        """
        User: 'The battery keeps dying, and I need to charge it frequently.'
        - Battery symptom: frequent discharge.
        - Does NOT fabricate a confirmed categorical condition.
        """
        state = AgentState(conversation_id="test_ex_c")
        user_msg = "The battery keeps dying, and I need to charge it frequently."
        ext = extract_explicit_facts(user_msg)

        assert "battery" in ext["symptoms"]
        assert "Battery_Status" not in [f["name"] for f in ext["extracted_features"]]

        apply_structured_extraction(state, ext, raw_user_message=user_msg)
        assert state.feature_state["Battery_Status"]["status"] == "unknown"
        assert state.symptoms.get("battery") is not None

    def test_example_d_unknown_engine_oil_change(self):
        """
        User: 'I don't know when I last changed the engine oil.'
        - Unconfirmed / unknown; does NOT invent dates or durations.
        """
        state = AgentState(conversation_id="test_ex_d")
        user_msg = "I don't know when I last changed the engine oil."
        ext = extract_explicit_facts(user_msg, current_question_feature="Maintenance_History")

        assert ext["extracted_features"] == []
        assert state.feature_state["Last_Service_Date"]["value"] is None
        assert "Maintenance_History" not in ext["unknown_features"]

    def test_example_e_multi_attribute_extraction(self):
        """
        User: 'My car is four years old, has run 55,000 kilometres, and I replaced the battery last year.'
        - Vehicle_Age = 4.
        - Mileage = 55,000 km recorded.
        - Battery replacement history recorded.
        """
        state = AgentState(conversation_id="test_ex_e")
        user_msg = "My car is four years old, has run 55,000 kilometres, and I replaced the battery last year."
        ext = extract_explicit_facts(user_msg)

        extracted_dict = {f["name"]: f["value"] for f in ext["extracted_features"]}
        assert extracted_dict.get("Vehicle_Age") == 4
        assert ext["unverified_unit_features"]["Mileage"]["value"] == 55000
        assert "battery" in ext["replacement_history"]

        apply_structured_extraction(state, ext, raw_user_message=user_msg)
        assert state.feature_state["Vehicle_Age"]["value"] == 4
        assert state.feature_state["Vehicle_Age"]["status"] == "known"
        assert "battery" in state.replacement_history


# ==============================================================================
# Model Schema & Question Budget Safety
# ==============================================================================

class TestSchemaAndQuestionBudget:

    def test_19_feature_schema_intact(self):
        """Ensure all 19 features exist with correct names in MODEL_FEATURE_SCHEMA."""
        expected_19 = {
            "Vehicle_Age", "Mileage", "Engine_Size", "Odometer_Reading",
            "Reported_Issues", "Service_History", "Accident_History",
            "Fuel_Efficiency", "Insurance_Premium", "Tire_Condition",
            "Brake_Condition", "Battery_Status", "Maintenance_History",
            "Fuel_Type", "Transmission_Type", "Owner_Type", "Vehicle_Model",
            "Last_Service_Date", "Warranty_Expiry_Date",
        }
        assert set(MODEL_FEATURE_SCHEMA) == expected_19

    def test_relative_date_answers_update_conversation_state(self):
        """
        Verify that relative date expressions update Last_Service_Date in AgentState,
        are marked as known, and advance the next-question selector.
        Supported test cases:
        - 'one year ago'
        - 'last year'
        - 'six months ago'
        - 'about two months back'
        - '25 November 2025'
        """
        from backend.app.agent.state.feature_selector import get_next_question_feature

        cases = [
            ("one year ago", True),
            ("last year", True),
            ("six months ago", True),
            ("about two months back", True),
            ("25 November 2025", False),
        ]
        for expr, is_approx in cases:
            state = AgentState(conversation_id=f"test_update_{expr.replace(' ', '_')}")
            state.current_question_feature = "Last_Service_Date"

            ext = extract_explicit_facts(expr, current_question_feature="Last_Service_Date")
            assert any(f["name"] == "Last_Service_Date" for f in ext["extracted_features"]), f"Extraction failed for: {expr}"
            extracted_val = next(f["value"] for f in ext["extracted_features"] if f["name"] == "Last_Service_Date")
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", extracted_val), f"Invalid date format '{extracted_val}' for: {expr}"

            apply_structured_extraction(state, ext, raw_user_message=expr)

            # Feature must be confirmed and known
            assert state.feature_state["Last_Service_Date"]["value"] == extracted_val
            assert state.feature_state["Last_Service_Date"]["status"] == "known"

            # Check approximation metadata
            if is_approx:
                assert "Last_Service_Date" in state.temporal_candidates
                assert state.temporal_candidates["Last_Service_Date"]["is_exact"] is False
                assert state.raw_user_inputs.get("approximate_features", {}).get("Last_Service_Date") == expr

            # Question selector must advance to next unanswered question (not repeat Last_Service_Date)
            next_q = get_next_question_feature(state)
            assert next_q != "Last_Service_Date", f"Question selector repeated Last_Service_Date for: {expr}"

    def test_negative_ambiguous_date_does_not_update_state(self):
        """Ambiguous expressions like 'not sure maybe sometime' must NOT update Last_Service_Date."""
        from backend.app.agent.state.feature_selector import get_next_question_feature

        state = AgentState(conversation_id="test_negative_ambig")
        state.current_question_feature = "Last_Service_Date"

        raw_msg = "not sure maybe sometime"
        ext = extract_explicit_facts(raw_msg, current_question_feature="Last_Service_Date")
        apply_structured_extraction(state, ext, raw_user_message=raw_msg)

        assert state.feature_state["Last_Service_Date"]["value"] is None
        assert state.feature_state["Last_Service_Date"]["status"] == "unknown"

    def test_standalone_last_service_relative_date_updates_state(self):
        """Standalone mention of last service date relative timing updates Last_Service_Date."""
        state = AgentState(conversation_id="test_standalone_service_date")
        raw_msg = "Last service was done about six months ago"
        ext = extract_explicit_facts(raw_msg)
        apply_structured_extraction(state, ext, raw_user_message=raw_msg)

        assert state.feature_state["Last_Service_Date"]["value"] is not None
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", state.feature_state["Last_Service_Date"]["value"])
        assert state.feature_state["Last_Service_Date"]["status"] == "known"
        assert "Last_Service_Date" in state.temporal_candidates

    def test_state_serialization_round_trip(self):
        """Ensure temporal_candidates and entity_observations survive serialization."""
        state = AgentState(conversation_id="test_round_trip")
        state.temporal_candidates["Last_Service_Date"] = {
            "approximate_date": "2024-11-22",
            "is_exact": False,
            "raw_text": "two years ago",
        }
        state.entity_observations["service_recency"] = "two years ago"

        serialized = state.to_json()
        restored = AgentState.from_json(serialized)

        assert restored.temporal_candidates["Last_Service_Date"]["raw_text"] == "two years ago"
        assert restored.entity_observations["service_recency"] == "two years ago"


# ==============================================================================
# Comprehensive NLP Intelligence Tests for Universal Understanding
# ==============================================================================

class TestTargetedNlpScenarios:
    """Validate specific NLP capabilities and failure scenarios defined in project requirements."""

    def test_scenario_1_relative_date_normalization(self):
        """Relative temporal expressions normalize to valid ISO date with approximation indicator."""
        test_cases = [
            ("one year ago", "year"),
            ("six months ago", "month"),
            ("two months back", "month"),
            ("three weeks ago", "week"),
            ("recently", "recent"),
            ("around November last year", "month_year"),
        ]
        for expr, expected_unit in test_cases:
            cand = parse_temporal_expressions_and_intervals(expr)
            assert cand is not None, f"Failed to parse temporal expression: {expr}"
            assert cand["is_exact"] is False, f"Approximation indicator not False for: {expr}"
            # Verify ISO format YYYY-MM-DD
            iso_date = cand["approximate_date"]
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", iso_date), f"Invalid ISO date '{iso_date}' for: {expr}"

    def test_scenario_2_last_year_i_did_timing_recognition(self):
        """'last year i did' under Maintenance_History recognizes service timing without guessing quality."""
        state = AgentState(conversation_id="test_last_year_timing")
        state.current_question_feature = "Maintenance_History"

        ext = extract_explicit_facts("last year i did", current_question_feature="Maintenance_History")
        assert any(a["type"] == "active_question_timing_clarification" for a in ext["ambiguities"])
        assert ext["service_recency"] == "last year"

        apply_structured_extraction(state, ext)
        assert state.feature_state["Maintenance_History"]["value"] is None
        assert state.active_ambiguity is not None
        assert state.active_ambiguity["recency"] == "last year"

    def test_scenario_3_battery_replacement_and_timing(self):
        """'i just changed my battery 2 months ago' records replacement history and timing without setting New."""
        ext = extract_explicit_facts("i just changed my battery 2 months ago")
        assert "battery" in ext["replacement_history"]
        assert ext["entity_observations"].get("battery_replacement_timing") == "2 months ago"
        assert not any(f["name"] == "Battery_Status" for f in ext["extracted_features"])

        state = AgentState(conversation_id="test_battery_repl")
        apply_structured_extraction(state, ext)
        assert state.feature_state["Battery_Status"]["value"] is None
        assert "battery" in state.replacement_history

    def test_scenario_4_subjective_vague_description_ambiguity(self):
        """'battery is kind of old' triggers ambiguity clarification and is NOT mapped to Weak."""
        ext = extract_explicit_facts("battery is kind of old", current_question_feature="Battery_Status")
        assert any(a["type"] == "subjective_condition_ambiguity" for a in ext["ambiguities"])
        assert not any(f["name"] == "Battery_Status" for f in ext["extracted_features"])

        state = AgentState(conversation_id="test_battery_ambig")
        state.current_question_feature = "Battery_Status"
        apply_structured_extraction(state, ext)
        assert state.feature_state["Battery_Status"]["value"] is None
        assert state.active_ambiguity is not None
        assert state.active_ambiguity["type"] == "subjective_condition_ambiguity"

        from backend.app.agent.adk_agents.judge_agent import generate_deterministic_fallback_response
        fallback = generate_deterministic_fallback_response(
            state, None, "Battery_Status", active_ambiguity=state.active_ambiguity
        )
        assert "New, Good, or Weak" in fallback

    def test_scenario_5_brake_working_properly(self):
        """'brake is working properly' and variants map to Brake_Condition = Good."""
        for phrase in ("brake is working properly", "brakes are working properly", "brakes work properly", "working well"):
            ext = extract_explicit_facts(phrase, current_question_feature="Brake_Condition")
            brake_feats = [f for f in ext["extracted_features"] if f["name"] == "Brake_Condition"]
            assert len(brake_feats) == 1, f"Failed for phrase: {phrase}"
            assert brake_feats[0]["value"] == "Good"

    def test_scenario_6_approximate_numerical_answers(self):
        """'approximately 5' extracts 5 with approximate indicator under numerical questions."""
        for feat in ("Service_History", "Reported_Issues", "Vehicle_Age"):
            ext = extract_explicit_facts("approximately 5", current_question_feature=feat)
            matched = [f for f in ext["extracted_features"] if f["name"] == feat]
            assert len(matched) == 1, f"Missing {feat}"
            assert matched[0]["value"] == 5
            assert feat in ext["approximate_features"]

    def test_scenario_7_no_accidents(self):
        """'no accidents' extracts Accident_History = 0."""
        ext = extract_explicit_facts("no accidents")
        acc = [f for f in ext["extracted_features"] if f["name"] == "Accident_History"]
        assert len(acc) == 1
        assert acc[0]["value"] == 0

    def test_scenario_8_component_age_never_sets_vehicle_age(self):
        """'The battery is almost 3 years old.' does NOT set Vehicle_Age = 3."""
        ext = extract_explicit_facts("The battery is almost 3 years old.")
        veh_age_feats = [f for f in ext["extracted_features"] if f["name"] == "Vehicle_Age"]
        assert len(veh_age_feats) == 0

    def test_scenario_9_engine_temperature_symptom(self):
        """'The engine temperature rises during long drives.' captures engine symptom."""
        ext = extract_explicit_facts("The engine temperature rises during long drives.")
        assert "engine" in ext["symptoms"]
        assert ext["user_intent"] == "overheating"

    def test_scenario_10_do_not_remember_service_date(self):
        """'I don't remember when I last serviced my car.' marks Last_Service_Date as unavailable."""
        ext = extract_explicit_facts("I don't remember when I last serviced my car.", current_question_feature="Last_Service_Date")
        assert "Last_Service_Date" in ext["unavailable_features"]

    def test_scenario_11_natural_language_feature_aliases(self):
        """Natural language aliases like 'distance driven' and 'number of services' are recognized."""
        is_v1, f1 = validate_feature_name("distance driven")
        assert is_v1 and f1 == "Mileage"
        is_v2, f2 = validate_feature_name("number of services")
        assert is_v2 and f2 == "Service_History"
        is_v3, f3 = validate_feature_name("number of accidents")
        assert is_v3 and f3 == "Accident_History"


# ==============================================================================
# Comprehensive NLP Improvements & Regression Verification
# ==============================================================================

class TestComprehensiveNlpImprovements:

    def test_end_to_end_five_turn_conversation_progression(self):
        """
        Simulate the exact requested 5-turn multi-turn conversation:
        Turn 1: "My car is 5 years old and I have noticed 3 problems recently."
        Turn 2: "It's fine only." (under Maintenance_History)
        Turn 3: "4 I guess." (under Service_History)
        Turn 4: "Serviced one year ago." (under Last_Service_Date)
        Turn 5: "Brake" (under Brake_Condition)
        Verify extraction, state confirmation, and progression at every turn.
        """
        from backend.app.agent.state.feature_selector import get_askable_features

        state = AgentState(conversation_id="test_five_turn_progression")
        ref_dt = datetime(2026, 11, 24, 12, 0, 0)
        history = []

        # Turn 1: User provides vehicle age and reported issues
        t1_msg = "My car is 5 years old and I have noticed 3 problems recently."
        ext1 = extract_explicit_facts(t1_msg, reference_date=ref_dt)
        apply_structured_extraction(state, ext1, raw_user_message=t1_msg)
        assert state.feature_state["Vehicle_Age"]["value"] == 5
        assert state.feature_state["Reported_Issues"]["value"] == 3
        history.append({"role": "user", "content": t1_msg})
        history.append({"role": "assistant", "content": "How would you describe your vehicle's maintenance history — Good, Average, or Poor?"})

        # Turn 2: Assistant asks Maintenance_History; User: "It's fine only."
        q1 = "Maintenance_History"
        t2_msg = "It's fine only."
        ext2 = extract_explicit_facts(t2_msg, current_question_feature=q1, conversation_history=history, reference_date=ref_dt)
        apply_structured_extraction(state, ext2, raw_user_message=t2_msg)
        assert state.feature_state["Maintenance_History"]["value"] == "Good"
        assert state.feature_state["Maintenance_History"]["status"] == "known"
        assert "Maintenance_History" not in get_askable_features(state)
        history.append({"role": "user", "content": t2_msg})
        history.append({"role": "assistant", "content": "How many times has your vehicle been serviced?"})

        # Turn 3: Assistant asks Service_History; User: "4 I guess."
        q2 = "Service_History"
        t3_msg = "4 I guess."
        ext3 = extract_explicit_facts(t3_msg, current_question_feature=q2, conversation_history=history, reference_date=ref_dt)
        apply_structured_extraction(state, ext3, raw_user_message=t3_msg)
        assert state.feature_state["Service_History"]["value"] == 4
        assert state.feature_state["Service_History"]["status"] == "known"
        assert "Service_History" not in get_askable_features(state)
        history.append({"role": "user", "content": t3_msg})
        history.append({"role": "assistant", "content": "When was your vehicle's last service?"})

        # Turn 4: Assistant asks Last_Service_Date; User: "Serviced one year ago."
        q3 = "Last_Service_Date"
        t4_msg = "Serviced one year ago."
        ext4 = extract_explicit_facts(t4_msg, current_question_feature=q3, conversation_history=history, reference_date=ref_dt)
        apply_structured_extraction(state, ext4, raw_user_message=t4_msg)
        assert state.feature_state["Last_Service_Date"]["value"] == "2025-11-24"
        assert state.feature_state["Last_Service_Date"]["status"] == "known"
        assert "Last_Service_Date" not in get_askable_features(state)
        history.append({"role": "user", "content": t4_msg})
        history.append({"role": "assistant", "content": "How would you rate the condition of your brakes?"})

        # Turn 5: Assistant asks Brake_Condition; User: "Brake"
        q4 = "Brake_Condition"
        t5_msg = "Brake"
        ext5 = extract_explicit_facts(t5_msg, current_question_feature=q4, conversation_history=history, reference_date=ref_dt)
        assert ext5["user_intent"] == "brake"
        # Brakes condition was not given a rating, so it remains askable
        assert "Brake_Condition" in get_askable_features(state)

    def test_relative_date_variations_for_last_service_date(self):
        """Test relative and exact dates when Last_Service_Date is active question."""
        from backend.app.agent.state.feature_selector import get_next_question_feature

        cases = [
            ("one year ago", "2025-11-24"),
            ("last year", "2025-11-24"),
            ("six months ago", "2026-05-24"),
            ("about two months back", "2026-09-24"),
            ("25 November 2025", "2025-11-25"),
        ]
        ref_dt = datetime(2026, 11, 24, 12, 0, 0)
        for phrase, expected_date in cases:
            state = AgentState(conversation_id=f"test_date_{phrase.replace(' ', '_')}")
            ext = extract_explicit_facts(phrase, current_question_feature="Last_Service_Date", reference_date=ref_dt)
            apply_structured_extraction(state, ext, raw_user_message=phrase)

            assert state.feature_state["Last_Service_Date"]["value"] == expected_date, f"Failed date for {phrase}"
            assert state.feature_state["Last_Service_Date"]["status"] == "known"
            next_feat = get_next_question_feature(state)
            assert next_feat != "Last_Service_Date", f"Failed to progress past Last_Service_Date for {phrase}"

    def test_negation_inversions(self):
        """Negations mapped to unambiguous opposite states where domain rules permit."""
        # Battery is not good -> Weak
        ext_bat = extract_explicit_facts("The battery is not good")
        extracted_bat = {f["name"]: f["value"] for f in ext_bat["extracted_features"]}
        assert extracted_bat.get("Battery_Status") == "Weak"

        # Brakes are not working properly -> Worn Out
        ext_brk = extract_explicit_facts("My brakes are not working properly")
        extracted_brk = {f["name"]: f["value"] for f in ext_brk["extracted_features"]}
        assert extracted_brk.get("Brake_Condition") == "Worn Out"

        # Tyres are not worn out -> Good
        ext_tyre = extract_explicit_facts("Tyres are not worn out")
        extracted_tyre = {f["name"]: f["value"] for f in ext_tyre["extracted_features"]}
        assert extracted_tyre.get("Tire_Condition") == "Good"

        # Engine is not overheating -> No symptom, user_intent not overheating
        ext_eng = extract_explicit_facts("The engine is not overheating")
        assert "engine" not in ext_eng["symptoms"]
        assert ext_eng["user_intent"] != "overheating"

    def test_combined_symptom_and_condition(self):
        """Simultaneous symptom and wear condition are both extracted."""
        ext = extract_explicit_facts("My brakes are squeaking and worn out")
        extracted = {f["name"]: f["value"] for f in ext["extracted_features"]}
        assert extracted.get("Brake_Condition") == "Worn Out"
        assert "brakes" in ext["symptoms"]
        assert "squeaking" in ext["symptoms"]["brakes"]

    def test_stated_distance_units(self):
        """Stated distance units with active question are parsed and added to extracted features."""
        # Under active question
        ext_active = extract_explicit_facts("around 60 thousand kilometres", current_question_feature="Mileage")
        extracted_active = {f["name"]: f["value"] for f in ext_active["extracted_features"]}
        assert extracted_active.get("Mileage") == 60000

        # Bare informal 60k
        ext_bare = extract_explicit_facts("I have driven around 60k")
        extracted_bare = {f["name"]: f["value"] for f in ext_bare["extracted_features"]}
        assert extracted_bare.get("Mileage") == 60000

    def test_exception_phrase_handling(self):
        """'Everything is fine except the battery' captures general status and notes exception."""
        ext = extract_explicit_facts("Everything is fine except the battery")
        extracted = {f["name"]: f["value"] for f in ext["extracted_features"]}
        assert extracted.get("Reported_Issues") == 1
        assert any("battery" in obs for obs in ext["raw_observations"])

    def test_never_serviced_handling(self):
        """'never serviced' under Last_Service_Date sets Service_History=0 and marks Last_Service_Date unavailable."""
        from backend.app.agent.state.feature_selector import get_next_question_feature

        state = AgentState(conversation_id="test_never_serviced")
        ext = extract_explicit_facts("never serviced", current_question_feature="Last_Service_Date")
        apply_structured_extraction(state, ext, raw_user_message="never serviced")

        assert state.feature_state["Service_History"]["value"] == 0
        assert state.feature_state["Service_History"]["status"] == "known"
        assert state.feature_state["Last_Service_Date"]["status"] == "unavailable"
        next_feat = get_next_question_feature(state)
        assert next_feat != "Last_Service_Date"


