"""
Feature Tools — Deterministic Python validation and state updates.
Validates LLM extractions against dataset schema, enforces canonical values,
resolves current question context, tolerates spelling errors, parses dates,
and prevents unconfirmed values from entering state.
"""

from __future__ import annotations

import re
import difflib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dateutil.relativedelta import relativedelta

try:
    from rapidfuzz import fuzz, process
except ImportError:
    # Graceful fallback to Python stdlib difflib if rapidfuzz is not installed
    class _FuzzFallback:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100.0

    class _ProcessFallback:
        @staticmethod
        def extractOne(query: str, choices, scorer=None, score_cutoff: float = 0):
            scorer = scorer or _FuzzFallback.ratio
            best_choice, best_score = None, -1.0
            for choice in choices:
                score = scorer(query, choice)
                if score > best_score:
                    best_choice, best_score = choice, score
            if best_choice is not None and best_score >= score_cutoff:
                return best_choice, best_score
            return None

    fuzz = _FuzzFallback()
    process = _ProcessFallback()

from backend.app.ml.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_VALUES,
    FEATURE_DISPLAY_NAMES,
)
from backend.app.agent.state.state import AgentState
from backend.app.ml.feature_schema import MODEL_FEATURE_SCHEMA as SELECTED_FEATURES, DATE_FEATURES, is_valid_feature
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_DECLINED_PATTERNS = (
    "don't want to share", "dont want to share", "do not want to share",
    "prefer not to say", "prefer not to share", "prefer not to share that",
    "prefer not sharing", "i prefer not to say", "i prefer not to share",
    "private", "refuse", "skip", "skip this",
    "none of your business", "wont say", "won't say", "do not want to tell",
    "don't want to tell", "dont want to tell", "decline to answer",
    "i don't want to share", "i do not want to share", "not sharing",
)

_UNAVAILABLE_PATTERNS = (
    "can't remember", "cant remember", "cannot remember", "dont remember",
    "don't remember", "do not remember", "i do not remember", "forgot", "forgotten",
    "can't recall", "cant recall", "cannot recall", "dont recall", "don't recall",
    "do not recall", "i do not recall",
    "i cannot remember", "i can't remember", "i cant remember", "i cannot recall", "i can't recall",
)

_UNKNOWN_PATTERNS = (
    "i do not know", "do not know", "i don't know", "i dont know", "dont know", "don't know",
    "not sure", "no idea", "no clue", "unclear", "not known", "unknown", "unsure",
    "i am unsure", "i'm unsure", "have no clue", "i have no clue", "have no idea",
    "haven't checked", "havent checked",
)

_PENDING_PATTERNS = (
    "i'll check and tell you", "ill check and tell you", "will check and tell you",
    "let me check and tell you", "will check", "i'll check", "ill check",
    "let me check", "let me find", "have to check", "i will look", "i'll look",
    "will let you know", "give me a moment", "checking right now", "need to check", "will look up",
)

_NEGATION_WORDS = {
    "not", "never", "isn't", "isnt", "aren't", "arent", "neither", "no", "without",
    "hardly", "barely", "cannot", "can't", "cant", "doesn't", "doesnt", "don't", "dont",
}

_APPROXIMATE_PATTERNS = (
    "maybe around", "around", "approx", "approximately", "about", "roughly",
    "give or take", "plus minus", "more or less",
)

_CORRECTION_PATTERNS = (
    "actually", "sorry meant", "correction", "i made a mistake", "i mean",
    "it's actually", "its actually", "wait, it is", "wait it is", "my mistake",
    "mistake,", "i meant",
)

_SUMMARY_YES_PATTERNS = (
    "yes", "yeah", "yep", "sure", "please", "yes please", "yes explain", "explain please",
    "show me", "give me the summary", "okay", "ok", "i'd like that", "show summary", "summary",
    "yes summary", "yes tell me", "tell me", "explain", "details", "yes i would like", "yes i would like to see it",
    "i would like to", "i'd like to", "yes please explain"
)

_SUMMARY_NO_PATTERNS = (
    "no", "no thanks", "not now", "that's enough", "i'm good", "nope", "no thank you",
    "nah", "dont need it", "don't need it", "stop"
)

_CANONICAL_DOMAIN_TARGETS = [
    "battery", "brakes", "tires", "engine", "transmission", "steering", "exhaust",
    "service", "serviced", "maintenance", "replaced", "changed",
    "year", "years", "month", "months", "week", "weeks", "ago",
    "good", "average", "poor", "worn", "weak",
]

_COMMON_TYPO_MAP = {
    # Components & Systems
    "batteru": "battery",
    "battry": "battery",
    "batry": "battery",
    "braks": "brakes",
    "brak": "brake",
    "brackes": "brakes",
    "braking": "braking",
    "tyrs": "tires",
    "tirs": "tires",
    "engne": "engine",
    "engn": "engine",
    "transmision": "transmission",
    "transmison": "transmission",
    "stering": "steering",
    "exaust": "exhaust",
    "temprature": "temperature",
    # Service & Maintenance actions
    "servisd": "serviced",
    "servce": "service",
    "maintanence": "maintenance",
    "maintainance": "maintenance",
    "maintenence": "maintenance",
    "replacd": "replaced",
    "repalced": "replaced",
    "chnged": "changed",
    # Distance & Metrics
    "miledge": "mileage",
    "odometre": "odometer",
    "odometor": "odometer",
    # Time units & adverbs
    "yer": "year",
    "yers": "years",
    "mnth": "month",
    "mnths": "months",
    "mths": "months",
    "agoo": "ago",
    "aproximately": "approximately",
    "aprox": "approx",
    # Categorical ratings & symptoms
    "god": "good",
    "gud": "good",
    "goof": "good",
    "avrage": "average",
    "por": "poor",
    "wek": "weak",
    "squeeking": "squeaking",
}

def normalize_typos_and_informal_tokens(text: str) -> str:
    """
    Produce a typo-normalized shadow copy of the text using rapidfuzz token similarity.
    Does NOT mutate the original user text string.
    Only corrects genuine typos; strictly preserves valid domain terms, syntax, and phrasing.
    """
    if not text:
        return text
    words = text.split()
    normalized = []
    for w in words:
        clean_w = w.lower().strip(".,!?;:()\"'")
        if not clean_w:
            normalized.append(w)
            continue
        # Direct typo lookup
        if clean_w in _COMMON_TYPO_MAP:
            normalized.append(_COMMON_TYPO_MAP[clean_w])
            continue
        # If it's already a recognized target or short word, keep it unchanged
        if clean_w in _CANONICAL_DOMAIN_TARGETS or len(clean_w) < 4:
            normalized.append(w)
            continue
        # Fuzzy match against known domain targets with a strict threshold (score >= 84)
        match = process.extractOne(clean_w, _CANONICAL_DOMAIN_TARGETS, scorer=fuzz.ratio, score_cutoff=84)
        if match:
            normalized.append(match[0])
            continue
        normalized.append(w)
    return " ".join(normalized)


_PRONOUN_PATTERN = re.compile(
    r"\b(it|that|they|those|the\s+component|the\s+part|the\s+previous\s+issue|that\s+component)\b",
    re.IGNORECASE
)

_COMPONENT_KEYWORDS = {
    "battery": ("battery", "batteries"),
    "brakes": ("brake", "brakes", "pads", "rotors", "calipers"),
    "tires": ("tire", "tires", "tyre", "tyres"),
    "engine": ("engine", "motor"),
    "transmission": ("transmission", "gearbox", "clutch"),
    "steering": ("steering", "wheel"),
    "exhaust": ("exhaust", "muffler"),
    "service": ("service", "maintenance", "servicing", "oil change"),
}

def resolve_contextual_antecedents(
    text: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    current_question_feature: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve anaphoric pronouns (it, that, they, the component) to the vehicle component
    mentioned in recent conversation history or the active question.
    Returns the resolved component name (e.g. 'battery', 'brakes', 'tires') or 'AMBIGUOUS' or None.
    """
    if not text:
        return None
    lower = text.lower()
    if not _PRONOUN_PATTERN.search(lower):
        return None

    # Check if text already explicitly names a component
    for comp, aliases in _COMPONENT_KEYWORDS.items():
        if any(a in lower for a in aliases):
            return comp

    # First check current_question_feature
    if current_question_feature:
        cq = current_question_feature.lower()
        if "battery" in cq:
            return "battery"
        elif "brake" in cq:
            return "brakes"
        elif "tire" in cq or "tyre" in cq:
            return "tires"
        elif "engine" in cq:
            return "engine"
        elif "transmission" in cq:
            return "transmission"
        elif "service" in cq or "maintenance" in cq:
            return "service"

    # Search backward in conversation history
    if conversation_history:
        for msg in reversed(conversation_history):
            content = (msg.get("content") or "").lower()
            found = []
            for comp, aliases in _COMPONENT_KEYWORDS.items():
                if any(a in content for a in aliases):
                    found.append(comp)
            if len(found) == 1:
                return found[0]
            elif len(found) > 1:
                return "AMBIGUOUS"

    return None


def parse_temporal_expressions_and_intervals(
    text: str,
    reference_date: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse relative temporal expressions (e.g. 'two years ago', 'six months earlier',
    'last week', 'recently', 'around Diwali') using runtime reference date.
    Calculates candidate date intervals without converting to a synthetic exact calendar date.
    """
    if not text:
        return None
    ref_date = reference_date or datetime.now()
    lower = text.lower()

    # Match month name + relative year: e.g. "around November last year", "in August this year", "November 2025"
    _MONTH_NAME_MAP = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    m_month_year = re.search(
        r"\b(?:around|in|about|near)?\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?:of\s+)?(last\s+year|this\s+year|\d{4})\b",
        lower
    )
    if m_month_year:
        m_name = m_month_year.group(1)
        m_num = _MONTH_NAME_MAP.get(m_name, 11)
        y_str = m_month_year.group(2)
        if "last" in y_str:
            t_year = ref_date.year - 1
        elif "this" in y_str:
            t_year = ref_date.year
        else:
            t_year = int(y_str)
        approx_str = f"{t_year}-{m_num:02d}-01"
        return {
            "raw": m_month_year.group(0).strip(),
            "raw_text": m_month_year.group(0).strip(),
            "approximate_date": approx_str,
            "candidate_start": approx_str,
            "candidate_end": f"{t_year}-{m_num:02d}-28",
            "is_exact": False,
            "is_inferred": True,
            "unit": "month_year",
            "quantity": 1,
        }

    # Match year expressions: "one year ago", "2 years ago", "two years back", "a year ago", "last year", "last yr"
    m_year = re.search(r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?)\s*(?:ago|back|earlier)?\b|\b(?:a|an|last)\s+(?:year|yr)\s*(?:ago|back|earlier)?\b", lower)
    if m_year:
        val_str = m_year.group(1) if m_year.group(1) else "1"
        num = parse_numerical_value(val_str) or 1
        d = ref_date - relativedelta(years=int(round(num)))
        return {
            "raw": m_year.group(0).strip(),
            "raw_text": m_year.group(0).strip(),
            "approximate_date": d.strftime("%Y-%m-%d"),
            "candidate_start": d.replace(day=1).strftime("%Y-%m-%d"),
            "candidate_end": d.replace(day=28).strftime("%Y-%m-%d"),
            "is_exact": False,
            "is_inferred": True,
            "unit": "year",
            "quantity": num,
        }

    # Match month expressions: "six months earlier", "a few months ago", "3 months ago", "last month"
    m_month = re.search(r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:months?|mnths?)\s*(?:ago|back|earlier)?\b|\b(?:a|an|last)\s+month\s*(?:ago|back|earlier)?\b|\b(?:a\s+few|a\s+couple(?:\s+of)?)\s+months\s*(?:ago|back|earlier)?\b", lower)
    if m_month:
        val_str = m_month.group(1) if m_month.group(1) else ("2" if "couple" in m_month.group(0) else ("3" if "few" in m_month.group(0) else "1"))
        num = parse_numerical_value(val_str) or 1
        d = ref_date - relativedelta(months=int(round(num)))
        return {
            "raw": m_month.group(0).strip(),
            "raw_text": m_month.group(0).strip(),
            "approximate_date": d.strftime("%Y-%m-%d"),
            "candidate_start": d.replace(day=1).strftime("%Y-%m-%d"),
            "candidate_end": d.replace(day=28).strftime("%Y-%m-%d"),
            "is_exact": False,
            "is_inferred": True,
            "unit": "month",
            "quantity": num,
        }

    # Match week expressions: "last week", "two weeks ago"
    m_week = re.search(r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*weeks?\s*(?:ago|back|earlier)?\b|\blast\s+week\b", lower)
    if m_week:
        val_str = m_week.group(1) if m_week.group(1) else "1"
        num = parse_numerical_value(val_str) or 1
        d = ref_date - relativedelta(weeks=int(round(num)))
        return {
            "raw": m_week.group(0).strip(),
            "raw_text": m_week.group(0).strip(),
            "approximate_date": d.strftime("%Y-%m-%d"),
            "candidate_start": d.strftime("%Y-%m-%d"),
            "candidate_end": d.strftime("%Y-%m-%d"),
            "is_exact": False,
            "is_inferred": True,
            "unit": "week",
            "quantity": num,
        }

    # Match days expressions: "a couple of days back", "few days ago"
    m_days = re.search(r"\b(\d+)\s*days?\s*(?:ago|back)?\b|\b(?:a\s+few|a\s+couple(?:\s+of)?)\s+days\s*(?:ago|back)?\b", lower)
    if m_days:
        val_str = m_days.group(1) if m_days.group(1) else "2"
        num = parse_numerical_value(val_str) or 2
        d = ref_date - relativedelta(days=int(num))
        return {
            "raw": m_days.group(0).strip(),
            "raw_text": m_days.group(0).strip(),
            "approximate_date": d.strftime("%Y-%m-%d"),
            "candidate_start": d.strftime("%Y-%m-%d"),
            "candidate_end": d.strftime("%Y-%m-%d"),
            "is_exact": False,
            "is_inferred": True,
            "unit": "day",
            "quantity": num,
        }

    # Festive / seasonal: "around diwali"
    if "diwali" in lower:
        year = ref_date.year if ref_date.month >= 10 else ref_date.year - 1
        return {
            "raw": "around Diwali",
            "raw_text": "around Diwali",
            "approximate_date": f"{year}-11-01",
            "candidate_start": f"{year}-10-15",
            "candidate_end": f"{year}-11-20",
            "is_exact": False,
            "is_inferred": True,
            "unit": "seasonal",
            "quantity": 1,
        }

    # Recent expressions: "recently", "just now", "lately"
    if any(w in lower for w in ("recently", "recent", "just now", "lately")):
        d = ref_date - relativedelta(months=1)
        return {
            "raw": "recently",
            "raw_text": "recently",
            "approximate_date": d.strftime("%Y-%m-%d"),
            "candidate_start": d.strftime("%Y-%m-%d"),
            "candidate_end": ref_date.strftime("%Y-%m-%d"),
            "is_exact": False,
            "is_inferred": True,
            "unit": "recent",
            "quantity": 1,
        }

    # Schedule / Cadence: "every six months"
    m_cadence = re.search(r"\bevery\s+(\d+|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(months?|years?)\b", lower)
    if m_cadence:
        raw_val = m_cadence.group(0).strip()
        return {
            "raw": raw_val,
            "raw_text": raw_val,
            "approximate_date": None,
            "is_cadence": True,
            "cadence_unit": m_cadence.group(2),
            "is_exact": False,
            "is_inferred": True,
        }

    # "since last year" / "hasn't been serviced since last year"
    if "since last year" in lower or "hasn't been serviced since last year" in lower or "havent been serviced since last year" in lower:
        prior_year = ref_date.year - 1
        return {
            "raw": "since last year",
            "raw_text": "since last year",
            "approximate_date": f"{prior_year}-06-01",
            "candidate_start": f"{prior_year}-01-01",
            "candidate_end": f"{prior_year}-12-31",
            "is_exact": False,
            "is_inferred": True,
            "unit": "year",
            "quantity": 1,
        }

    return None

_SERVICE_TIMING_PATTERN = re.compile(
    r"\b(?:(?:it|that)\s+(?:was|got|is)?\s*(?:done|serviced|maintained)|"
    r"(?:maintenance|service)\s+(?:was\s+)?(?:done|performed|carried\s+out)|"
    r"had\s+it\s+(?:done|serviced)|"
    r"i\s+(?:did(?:\s+it)?|had\s+it\s+done)\s*(?:about\s+|around\s+)?|"
    r"(?:serviced|done))\s*(?:about\s+|around\s+)?"
    r"(recently|just\s+now|last\s+(?:year|month|week)|\d+(?:\.\d+)?\s*(?:years?|yrs?|months?|weeks?)\s*ago|"
    r"\b(?:one|two|three|four|five|six|half\s+a\s+dozen|couple(?:\s+of)?|few|a\s+few)\b\s*(?:years?|months?|weeks?)\s*ago|"
    r"(?:around\s+|in\s+|about\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?:of\s+)?(?:last\s+year|this\s+year|\d{4})|"
    r"(?:a|an)\s+(?:year|month|week)\s+ago)\b|"
    r"\b(recently|just\s+now|last\s+(?:year|month|week)|\d+(?:\.\d+)?\s*(?:years?|yrs?|months?|weeks?)\s*ago|"
    r"\b(?:one|two|three|four|five|six|half\s+a\s+dozen|couple(?:\s+of)?|few|a\s+few)\b\s*(?:years?|months?|weeks?)\s*ago|"
    r"(?:around\s+|in\s+|about\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?:of\s+)?(?:last\s+year|this\s+year|\d{4})|"
    r"(?:a|an)\s+(?:year|month|week)\s+ago)\s+(?:i\s+did(?:\s+it)?|it\s+was\s+done|was\s+the\s+last\s+time)\b|"
    r"^\s*(?:about\s+|around\s+)?(recently|just\s+now|last\s+(?:year|month|week)|\d+(?:\.\d+)?\s*(?:years?|yrs?|months?|weeks?)\s*ago|"
    r"\b(?:one|two|three|four|five|six|half\s+a\s+dozen|couple(?:\s+of)?|few|a\s+few)\b\s*(?:years?|months?|weeks?)\s*ago|"
    r"(?:around\s+|in\s+|about\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?:of\s+)?(?:last\s+year|this\s+year|\d{4})|"
    r"(?:a|an)\s+(?:year|month|week)\s+ago)\.?\s*$",
    re.IGNORECASE
)

def parse_relative_service_timing(text: str) -> Optional[str]:
    """
    Parse relative service timing expressions (e.g. 'a year ago', 'last year', 'six months ago', 'recently').
    Does NOT convert to an exact calendar date.
    """
    if not text:
        return None
    m = _SERVICE_TIMING_PATTERN.search(text)
    if m:
        recency = next((g for g in m.groups() if g is not None), None)
        if recency:
            return recency.strip().rstrip(".")
    return None


def classify_post_assessment_intent(user_message: str, previous_mode: str = "AWAITING_SUMMARY_CONFIRMATION") -> str:
    """
    Classify user intent during post-assessment conversational turns.
    Returns:
      "SUMMARY_REQUEST"
      "SUMMARY_DECLINE"
      "FACTOR_EXPLANATION"
      "RISK_EXPLANATION"
      "ASSESSMENT_DETAILS"
      "RECOMMENDATION"
      "CAPABILITY_QUESTION"
      "GENERAL_FOLLOW_UP"
    """
    clean = re.sub(r"[^\w\s]", " ", (user_message or "").lower()).strip()
    clean_compact = " ".join(clean.split())
    if not clean_compact:
        return "GENERAL_FOLLOW_UP"

    # An explicit request to start over is the only post-assessment request
    # allowed to clear the canonical assessment facts.
    if any(p in clean_compact for p in (
        "new assessment", "start a new assessment", "start new assessment",
        "start over", "restart assessment", "another assessment", "assess another"
    )):
        return "NEW_ASSESSMENT"

    # 1. Capability Boundaries (RUL, Causal claims)
    if any(p in clean_compact for p in (
        "remaining useful life", "useful life", "years left", "how many years", "remaining life",
        "caused the failure", "cause of failure", "why did it fail", "failure cause", "root cause",
        "what caused"
    )):
        return "CAPABILITY_QUESTION"

    # 2. Factor Explanation (What is the main factor / what affected the score / what drove the score)
    if any(p in clean_compact for p in (
        "main factor", "top factor", "most important factor", "strongest factor",
        "which factor", "what factor", "what affected", "what influenced", "which feature",
        "what contributed", "factor affected", "factor mattered", "factors behind", "which one",
        "what drove", "affected the score", "affected score", "affected my score",
        "what changed the score", "what makes the score", "what caused the score",
        "primary factor", "leading factor", "key factor", "why is this factor important",
        "why did this affect", "why this feature", "explain this factor"
    )):
        return "FACTOR_EXPLANATION"

    # 3. Risk Explanation (Why is the risk low / high / what is the reason for this output)
    if any(p in clean_compact for p in (
        "why is the risk", "why is risk", "why low", "why high", "why did it predict",
        "reason for risk", "reason for this output", "reason for this result",
        "reason for the output", "reason for the result", "reason for my result",
        "why this output", "why this result", "why the score", "why is it low",
        "why is it high", "why is my risk", "what is the reason", "what's the reason",
        "why is the score", "why the prediction", "why maintenance required",
        "why did i get this result", "why did i get this", "why this result",
        "why is my result", "why did i receive this", "what caused this result",
        "why is my score", "why is it showing", "why is the output"
    )) or (clean_compact in ("why", "why?", "how so", "how come", "why this?")):
        return "RISK_EXPLANATION"


    # Short replies are meaningful after a result has been shown.  They should
    # continue the result discussion rather than fall through to the completed
    # assessment closing prompt.
    if previous_mode in ("SUMMARY", "EXPLANATION", "FOLLOW_UP") and clean_compact in (
        "explain", "explain it", "tell me", "how", "how?", "what affected it"
    ):
        return "RISK_EXPLANATION"

    # 4. Assessment Details / Features Used
    if any(p in clean_compact for p in (
        "what information did you use", "what details did you use", "what data did you use",
        "what features were used", "what did you base"
    )):
        return "ASSESSMENT_DETAILS"

    # 5. Recommendation / Next Steps
    if any(p in clean_compact for p in (
        "what should i do", "what should i check", "what to check", "check first",
        "reduce risk", "lower risk", "next steps", "recommendation", "what now", "what next"
    )):
        return "RECOMMENDATION"

    # 6. Decline
    if clean_compact in _SUMMARY_NO_PATTERNS or any(clean_compact == p for p in _SUMMARY_NO_PATTERNS) or any(clean_compact.startswith(p) for p in ("no ", "not now", "nope", "nah")):
        return "SUMMARY_DECLINE"

    # 7. Summary Confirmation / Request
    if previous_mode == "AWAITING_SUMMARY_CONFIRMATION":
        if clean_compact in _SUMMARY_YES_PATTERNS or any(clean_compact == p for p in _SUMMARY_YES_PATTERNS) or any(clean_compact.startswith(p) for p in ("yes", "yeah", "yep", "sure", "please", "show", "give", "explain", "i'd like")):
            return "SUMMARY_REQUEST"
    else:
        if any(p in clean_compact for p in ("summary", "recap", "give me summary", "show summary", "see summary")):
            return "SUMMARY_REQUEST"

    return "GENERAL_FOLLOW_UP"


def classify_summary_intent(user_message: str) -> Optional[str]:
    """Backward-compatible helper returning 'yes' or 'no'."""
    intent = classify_post_assessment_intent(user_message, "AWAITING_SUMMARY_CONFIRMATION")
    if intent == "SUMMARY_REQUEST":
        return "yes"
    if intent == "SUMMARY_DECLINE":
        return "no"
    return None

_NUMBER_WORDS = {
    "zero": 0, "none": 0, "no": 0, "nil": 0,
    "one": 1, "single": 1, "a": 1, "an": 1,
    "two": 2, "couple": 2, "a couple": 2, "a couple of": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "half a dozen": 6,
    "fifteen": 15, "twenty": 20, "twenty five": 25, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

_MULTIPLIERS: Dict[str, int] = {
    "k": 1_000,
    "kilo": 1_000,
    "thousand": 1_000,
    "thousands": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "millions": 1_000_000,
}

def parse_numerical_value(raw_val: Any) -> Optional[int | float]:
    """
    Parse natural language numerical strings into non-negative integers or floats.
    Handles number words ('three', 'couple', 'dozen'), multipliers ('50k', '1.5 lakh', '2 million'),
    and commas in numbers ('50,000').
    """
    if raw_val is None:
        return None
    if isinstance(raw_val, (int, float)):
        val = float(raw_val)
        if val < 0:
            return None
        return int(val) if val.is_integer() else val

    val_str = str(raw_val).strip().lower().replace(",", "")
    if not val_str:
        return None

    # 1. Exact number word match
    if val_str in _NUMBER_WORDS:
        return _NUMBER_WORDS[val_str]

    # 2. Match compound word + multiplier, e.g. "sixty thousand", "two lakh", "one million"
    word_mult_pattern = r"\b(" + "|".join(re.escape(w) for w in _NUMBER_WORDS.keys()) + r")\s*(k|kilo|thousand|thousands|lakh|lakhs|lac|lacs|m|million|millions|hundred|hundreds)\b"
    wm = re.search(word_mult_pattern, val_str)
    if wm:
        num_part = float(_NUMBER_WORDS[wm.group(1)])
        mult_key = wm.group(2).lower()
        mult_part = 100 if "hundred" in mult_key else _MULTIPLIERS.get(mult_key, 1)
        res = num_part * mult_part
        return int(res) if res.is_integer() else res

    # 3. Match number with multiplier (e.g., '50k', '1.5 lakh', '2.5 million', 'around 60k')
    mult_pattern = r"\b(\d+(?:\.\d+)?)\s*(k|kilo|thousand|thousands|lakh|lakhs|lac|lacs|m|million|millions)\b"
    m = re.search(mult_pattern, val_str)
    if m:
        num_part = float(m.group(1))
        mult_part = _MULTIPLIERS.get(m.group(2).lower(), 1)
        res = num_part * mult_part
        return int(res) if res.is_integer() else res

    # 4. Match pure digits / float
    digit_match = re.search(r"^(\d+(?:\.\d+)?)$", val_str)
    if digit_match:
        res = float(digit_match.group(1))
        return int(res) if res.is_integer() else res

    # 5. Search for digit in substring
    digit_sub = re.search(r"\b(\d+(?:\.\d+)?)\b", val_str)
    if digit_sub:
        res = float(digit_sub.group(1))
        return int(res) if res.is_integer() else res

    # 6. Search for number word in substring
    for nw, n_val in sorted(_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if nw in ("a", "an", "no", "single"):
            continue
        if re.search(rf"\b{re.escape(nw)}\b", val_str):
            return n_val

    return None


def parse_fuel_efficiency_with_unit(raw_val: Any) -> Optional[Dict[str, Any]]:
    """
    Parse fuel efficiency expression into value, unit, and whether it is ambiguous.
    Distinguishes:
    - km/L / kmpl
    - L/100km
    - US MPG (explicit)
    - Imperial MPG (explicit)
    - Generic MPG (ambiguous)
    - Unitless number
    """
    if raw_val is None:
        return None
    val_str = str(raw_val).strip().lower()

    # 1. Check L/100km (e.g., "7 L/100km", "7.5 litres per 100 km")
    l_100_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:l/100\s*km|litres?\s*per\s*100\s*km|l/100km)\b", val_str)
    if l_100_match:
        num = float(l_100_match.group(1))
        if num > 0:
            return {"value": num, "unit": "l/100km", "is_generic_mpg": False, "raw": val_str}

    # 2. Check Explicit US MPG
    us_mpg_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:us\s*mpg|us\s*miles?\s*per\s*gallon)\b", val_str)
    if us_mpg_match:
        return {"value": float(us_mpg_match.group(1)), "unit": "us_mpg", "is_generic_mpg": False, "raw": val_str}

    # 3. Check Explicit Imperial MPG / UK MPG
    uk_mpg_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:imperial\s*mpg|uk\s*mpg|imperial\s*miles?\s*per\s*gallon)\b", val_str)
    if uk_mpg_match:
        return {"value": float(uk_mpg_match.group(1)), "unit": "imperial_mpg", "is_generic_mpg": False, "raw": val_str}

    # 4. Check Generic MPG (ambiguous between US and Imperial)
    mpg_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mpg|miles?\s*per\s*gallon)\b", val_str)
    if mpg_match:
        return {"value": float(mpg_match.group(1)), "unit": "generic_mpg", "is_generic_mpg": True, "raw": val_str}

    # 5. Check km/L or kmpl
    kmpl_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km/?l|kmpl|kilometres?\s*per\s*litre)\b", val_str)
    if kmpl_match:
        return {"value": float(kmpl_match.group(1)), "unit": "km/l", "is_generic_mpg": False, "raw": val_str}

    # 6. Unitless number fallback
    num_val = parse_numerical_value(val_str)
    if num_val is not None and 1.0 <= float(num_val) <= 60.0:
        return {"value": float(num_val), "unit": "unitless", "is_generic_mpg": False, "raw": val_str}

    return None


def convert_fuel_efficiency(raw_val: Any) -> Optional[float]:
    """
    Parse fuel efficiency expressions without assuming canonical dataset units.
    Returns float value for unitless or standard metric (km/l, kmpl) inputs.
    For unit-stated inputs (MPG, L/100km), returns None to prevent unverified unit conversion into model state.
    """
    parsed = parse_fuel_efficiency_with_unit(raw_val)
    if not parsed:
        return None
    if parsed["unit"] in ("unitless", "km/l"):
        return round(float(parsed["value"]), 2)
    return None


_NEGATION_INVERSIONS: Dict[str, Dict[str, str]] = {
    "Brake_Condition": {
        "Worn Out": "Good",
        "Good": "Worn Out",
    },
    "Battery_Status": {
        "Good": "Weak",
    },
    "Tire_Condition": {
        "Worn Out": "Good",
        "Good": "Worn Out",
    },
    "Maintenance_History": {
        "Poor": "Good",
    },
}


def evaluate_component_condition(
    text: str,
    component_name: str,
    component_aliases: Tuple[str, ...],
    allowed_values: List[str],
    synonyms: Dict[str, str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], List[str], bool]:
    """
    Evaluate condition assertions for a component with full negation awareness.
    Returns:
      (confirmed_value, contradiction_dict, observations, is_negated_only)
    """
    from typing import Set
    lower = text.lower()

    # Tokenize words for negation tracking
    words = re.findall(r"\b[\w'-]+\b", lower)

    # Map each synonym or allowed value
    affirmed_conditions: Set[str] = set()
    negated_conditions: Set[str] = set()

    all_keys = sorted(list(synonyms.keys()) + [a.lower() for a in allowed_values], key=len, reverse=True)

    for key in all_keys:
        for match in re.finditer(rf"\b{re.escape(key)}\b", lower):
            start_pos = match.start()
            prefix_text = lower[:start_pos]
            prefix_tokens = re.findall(r"\b[\w'-]+\b", prefix_text)
            window = prefix_tokens[-4:] if len(prefix_tokens) >= 4 else prefix_tokens
            is_negated = any(tok in _NEGATION_WORDS for tok in window)

            canonical_val = synonyms.get(key, None)
            if not canonical_val:
                for a in allowed_values:
                    if a.lower() == key:
                        canonical_val = a
                        break
            if not canonical_val:
                continue

            if is_negated:
                negated_conditions.add(canonical_val)
            else:
                affirmed_conditions.add(canonical_val)

    if not affirmed_conditions and not negated_conditions:
        return None, None, [], False

    observations: List[str] = []

    # 1. Multiple affirmed conditions: check for conflict
    if len(affirmed_conditions) > 1:
        healthy = {"Good", "New"}
        degraded = {"Worn Out", "Poor", "Weak", "Average"}
        has_healthy = bool(affirmed_conditions & healthy)
        has_degraded = bool(affirmed_conditions & {"Worn Out", "Poor", "Weak"})
        if has_healthy and has_degraded:
            contra = {
                "feature": component_name,
                "conflicting_values": sorted(list(affirmed_conditions)),
                "raw": text,
            }
            obs = f"Contradiction detected in assertion for {component_name}: affirmed conflicting conditions {', '.join(sorted(affirmed_conditions))}. Clarification needed."
            observations.append(obs)
            return None, contra, observations, False
        if affirmed_conditions <= healthy:
            val = "New" if "New" in affirmed_conditions else "Good"
            return val, None, observations, False

    # 2. Single affirmed condition
    if len(affirmed_conditions) == 1:
        affirmed_val = next(iter(affirmed_conditions))
        if affirmed_val in negated_conditions:
            contra = {
                "feature": component_name,
                "conflicting_values": [affirmed_val, f"not {affirmed_val}"],
                "raw": text,
            }
            obs = f"Contradiction detected for {component_name}: affirmed {affirmed_val} but also negated {affirmed_val}."
            observations.append(obs)
            return None, contra, observations, False
        if negated_conditions:
            observations.append(f"User affirmed {component_name} is {affirmed_val} and confirmed not {', '.join(sorted(negated_conditions))}.")
        return affirmed_val, None, observations, False

    # 3. Negated conditions only (no affirmed conditions)
    if not affirmed_conditions and negated_conditions:
        if len(negated_conditions) == 1:
            neg_val = next(iter(negated_conditions))
            if component_name in _NEGATION_INVERSIONS and neg_val in _NEGATION_INVERSIONS[component_name]:
                inferred_val = _NEGATION_INVERSIONS[component_name][neg_val]
                obs = f"User negated {neg_val} for {component_name}; mapped to {inferred_val} based on domain negation."
                observations.append(obs)
                return inferred_val, None, observations, False
        observations.append(f"User negated {', '.join(sorted(negated_conditions))} for {component_name} (condition remains unconfirmed).")
        return None, None, observations, True

    return None, None, [], False


def convert_engine_size(raw_val: Any) -> Optional[int]:
    """
    Convert engine displacement to integer cc.
    - Litres (e.g., '1.5L', '1.5 litre', '2.0 liter') -> Litres * 1000
    - cc (e.g., '1500cc', '800 cc') -> integer cc
    """
    if raw_val is None:
        return None
    val_str = str(raw_val).strip().lower()

    # Litres match (e.g., '1.5L', '1.5 litre', '2.0 liter')
    litre_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:l|litre|liter|litres|liters)\b", val_str)
    if litre_match:
        litres = float(litre_match.group(1))
        if 0.5 <= litres <= 8.5:
            return int(round(litres * 1000))

    # cc match
    cc_match = re.search(r"(\d+)\s*(?:cc|cubic\s*centimet(?:er|re)s?)\b", val_str)
    if cc_match:
        return int(cc_match.group(1))

    # Raw number fallback
    parsed = parse_numerical_value(val_str)
    if parsed is not None:
        f_val = float(parsed)
        if 0.5 <= f_val <= 8.5:
            return int(round(f_val * 1000))
        elif 500 <= f_val <= 8500:
            return int(f_val)

    return None

_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y",
    "%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y",
    "%Y/%m/%d", "%m/%d/%Y",
)

# Canonical feature aliases and typo mappings
_FEATURE_NAME_ALIASES: Dict[str, str] = {
    # Brake_Condition
    "brake": "Brake_Condition", "brakes": "Brake_Condition", "brke": "Brake_Condition",
    "brkes": "Brake_Condition", "breaks": "Brake_Condition", "break": "Brake_Condition",
    "brake condition": "Brake_Condition", "brakes condition": "Brake_Condition",
    # Battery_Status
    "battery": "Battery_Status", "batteru": "Battery_Status", "batery": "Battery_Status",
    "battery status": "Battery_Status", "battery condition": "Battery_Status",
    # Maintenance_History
    "maintenance": "Maintenance_History", "maintainance": "Maintenance_History",
    "maintenence": "Maintenance_History", "maintanence": "Maintenance_History",
    "maintenance history": "Maintenance_History", "maintainance history": "Maintenance_History",
    # Service_History
    "service": "Service_History", "services": "Service_History", "service history": "Service_History",
    "service visits": "Service_History", "serviced": "Service_History", "servicing": "Service_History",
    "number of services": "Service_History", "service count": "Service_History", "past services": "Service_History",
    # Reported_Issues
    "reported issues": "Reported_Issues", "issues": "Reported_Issues", "problems": "Reported_Issues",
    "reported problems": "Reported_Issues", "faults": "Reported_Issues", "issues count": "Reported_Issues",
    "number of issues": "Reported_Issues", "issue count": "Reported_Issues",
    # Vehicle_Age
    "vehicle age": "Vehicle_Age", "age": "Vehicle_Age", "car age": "Vehicle_Age", "bike age": "Vehicle_Age",
    # Tire_Condition
    "tire": "Tire_Condition", "tires": "Tire_Condition", "tyre": "Tire_Condition", "tyres": "Tire_Condition",
    "tire condition": "Tire_Condition", "tyre condition": "Tire_Condition",
    # Accident_History
    "accident": "Accident_History", "accidents": "Accident_History", "accident history": "Accident_History",
    "crashes": "Accident_History", "number of accidents": "Accident_History", "accident count": "Accident_History",
    # Last_Service_Date
    "last service": "Last_Service_Date", "last service date": "Last_Service_Date",
    "last serviced": "Last_Service_Date", "service date": "Last_Service_Date",
    # Warranty_Expiry_Date
    "warranty": "Warranty_Expiry_Date", "warranty expiry": "Warranty_Expiry_Date",
    "warranty expiry date": "Warranty_Expiry_Date", "warranty date": "Warranty_Expiry_Date",
    # Numerical features
    "mileage": "Mileage", "odometer": "Odometer_Reading", "odometer reading": "Odometer_Reading", "odo": "Odometer_Reading",
    "distance driven": "Mileage", "distance travelled": "Mileage", "distance traveled": "Mileage", "distance": "Mileage",
    "kilometres run": "Mileage", "kilometers run": "Mileage", "kms driven": "Mileage", "miles driven": "Mileage",
    "engine size": "Engine_Size", "engine": "Engine_Size", "displacement": "Engine_Size",
    "fuel efficiency": "Fuel_Efficiency", "fuel economy": "Fuel_Efficiency", "mileage per litre": "Fuel_Efficiency",
    "insurance premium": "Insurance_Premium", "insurance": "Insurance_Premium", "insurance cost": "Insurance_Premium",
    # Categorical features
    "fuel type": "Fuel_Type", "fuel": "Fuel_Type", "transmission": "Transmission_Type",
    "transmission type": "Transmission_Type", "gearbox": "Transmission_Type",
    "owner type": "Owner_Type", "owner": "Owner_Type", "ownership": "Owner_Type",
    "vehicle model": "Vehicle_Model", "vehicle type": "Vehicle_Model", "model": "Vehicle_Model", "body type": "Vehicle_Model",
}

# Categorical synonyms and typo dictionary
_CATEGORICAL_SYNONYMS: Dict[str, Dict[str, str]] = {
    "Maintenance_History": {
        "good": "Good", "gud": "Good", "great": "Good", "well maintained": "Good", "excellent": "Good",
        "regular": "Good", "fine": "Good", "fine only": "Good", "it's fine only": "Good", "its fine only": "Good",
        "all good": "Good", "everything good": "Good", "everything fine": "Good", "everything is fine": "Good",
        "everything is good": "Good", "properly": "Good", "up to date": "Good", "dealer maintained": "Good",
        "full service history": "Good",
        "average": "Average", "avrage": "Average", "avg": "Average", "moderate": "Average", "ok": "Average",
        "okay": "Average", "fair": "Average", "normal": "Average", "decent": "Average", "medium": "Average",
        "poor": "Poor", "bad": "Poor", "neglected": "Poor", "terrible": "Poor", "rough": "Poor", "irregular": "Poor",
    },
    "Brake_Condition": {
        "new": "New", "brand new": "New", "brand new brakes": "New", "newly fitted pads": "New",
        "good": "Good", "gud": "Good", "fine": "Good", "fine only": "Good", "it's fine only": "Good", "its fine only": "Good",
        "ok": "Good", "okay": "Good", "decent": "Good", "no issues": "Good", "properly": "Good",
        "working": "Good", "working properly": "Good", "works properly": "Good", "work properly": "Good",
        "working well": "Good", "works well": "Good", "working fine": "Good", "works fine": "Good",
        "functioning properly": "Good", "functions properly": "Good",
        "great": "Good", "responsive": "Good", "healthy": "Good",
        "worn out": "Worn Out", "worn": "Worn Out", "wornout": "Worn Out", "badly worn": "Worn Out",
        "metal on metal": "Worn Out", "bald pads": "Worn Out",
    },
    "Battery_Status": {
        "new": "New", "brand new": "New", "brand new battery": "New", "fresh battery": "New",
        "good": "Good", "gud": "Good", "fine": "Good", "fine only": "Good", "it's fine only": "Good", "its fine only": "Good",
        "ok": "Good", "okay": "Good", "healthy": "Good", "no issues": "Good", "properly": "Good",
        "working": "Good", "working properly": "Good", "works properly": "Good", "work properly": "Good",
        "working well": "Good", "works well": "Good", "working fine": "Good", "works fine": "Good",
        "strong": "Good", "great": "Good", "holds charge": "Good",
        "weak": "Weak", "wek": "Weak", "failing health": "Weak", "degraded": "Weak", "low capacity": "Weak",
    },
    "Tire_Condition": {
        "new": "New", "brand new": "New", "brand new tires": "New", "new rubber": "New",
        "good": "Good", "gud": "Good", "fine": "Good", "fine only": "Good", "it's fine only": "Good", "its fine only": "Good",
        "ok": "Good", "okay": "Good", "decent": "Good", "no issues": "Good", "properly": "Good",
        "working": "Good", "working properly": "Good", "works properly": "Good", "work properly": "Good",
        "working well": "Good", "works well": "Good", "working fine": "Good", "works fine": "Good",
        "great": "Good", "plenty of tread": "Good",
        "worn out": "Worn Out", "worn": "Worn Out", "wornout": "Worn Out", "bald": "Worn Out",
        "smooth": "Worn Out", "tread depth low": "Worn Out", "cords showing": "Worn Out",
    },
    "Fuel_Type": {
        "petrol": "Petrol", "gasoline": "Petrol", "unleaded": "Petrol",
        "diesel": "Diesel", "tdi": "Diesel",
        "electric": "Electric", "ev": "Electric", "battery electric": "Electric",
    },
    "Transmission_Type": {
        "automatic": "Automatic", "auto": "Automatic", "at": "Automatic", "amt": "Automatic",
        "cvt": "Automatic", "dsg": "Automatic", "dct": "Automatic",
        "manual": "Manual", "stick": "Manual", "stick shift": "Manual", "mt": "Manual", "standard": "Manual",
    },
    "Owner_Type": {
        "first": "First", "1st": "First", "single owner": "First", "one owner": "First", "1 owner": "First",
        "original owner": "First", "bought new": "First",
        "second": "Second", "2nd": "Second", "two owners": "Second", "2 owners": "Second", "second hand": "Second",
        "third": "Third", "3rd": "Third", "three owners": "Third", "3 owners": "Third",
    },
    "Vehicle_Model": {
        "car": "Car", "sedan": "Car", "hatchback": "Car", "coupe": "Car", "wagon": "Car",
        "suv": "SUV", "crossover": "SUV",
        "truck": "Truck", "pickup": "Truck", "lorry": "Truck",
        "van": "Van", "minivan": "Van", "panel van": "Van",
        "motorcycle": "Motorcycle", "bike": "Motorcycle", "scooter": "Motorcycle", "two-wheeler": "Motorcycle",
        "bus": "Bus", "minibus": "Bus", "coach": "Bus",
    },
}


def normalise_date_value(raw_value: Any) -> Optional[str]:
    """Return a real calendar date as the model's ISO-8601 representation (YYYY-MM-DD)."""
    if not raw_value:
        return None
    value = str(raw_value).strip()
    # Strip ordinal suffixes (15th -> 15, 1st -> 1, 2nd -> 2, 3rd -> 3)
    value = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    # Remove leading date phrases like "last serviced on", "serviced on", "date:", etc.
    value = re.sub(r"^(?:last\s+serviced\s+(?:on\s+)?|serviced\s+(?:on\s+)?|last\s+service\s+(?:date\s+)?(?:is|=|:)?\s*|warranty\s+(?:expiry\s+)?(?:date\s+)?(?:is|=|:)?\s*|date\s*(?:is|=|:)?\s*)", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+", " ", value)

    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue

    # Try matching regex date within string
    patterns = [
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, value, re.IGNORECASE)
        if m:
            sub_val = m.group(1).replace(",", "")
            for date_format in _DATE_FORMATS:
                try:
                    return datetime.strptime(sub_val, date_format).date().isoformat()
                except ValueError:
                    continue
    return None


def validate_feature_name(feature_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate whether a feature belongs strictly to the dataset schema using central validation.
    Tolerates spelling mistakes, abbreviations, and aliases.
    Returns (is_valid, canonical_name).
    """
    if not feature_name:
        return False, None

    cleaned = str(feature_name).strip().lower().replace(" ", "_")

    # 1. Exact match against SELECTED_FEATURES
    for valid_feat in SELECTED_FEATURES:
        if valid_feat.lower() == cleaned or valid_feat.lower().replace("_", "") == cleaned.replace("_", ""):
            return True, valid_feat

    # 2. Check alias dictionary
    alias_key = str(feature_name).strip().lower().replace("_", " ")
    if alias_key in _FEATURE_NAME_ALIASES:
        canonical = _FEATURE_NAME_ALIASES[alias_key]
        if canonical in SELECTED_FEATURES:
            return True, canonical

    # 3. Fuzzy match against SELECTED_FEATURES and aliases (strict cutoff to avoid generic 'condition' false positives)
    candidates = list(SELECTED_FEATURES) + list(_FEATURE_NAME_ALIASES.keys())
    matches = difflib.get_close_matches(alias_key, candidates, n=1, cutoff=0.85)
    if matches:
        matched = matches[0]
        canonical = _FEATURE_NAME_ALIASES.get(matched, matched)
        if canonical in SELECTED_FEATURES:
            return True, canonical

    return False, None



def fuzzy_match_categorical(canonical_name: str, raw_value: Any) -> Optional[str]:
    """
    Match raw_value against allowed values for a categorical feature with spelling tolerance.
    """
    if raw_value is None or canonical_name not in CATEGORICAL_VALUES:
        return None

    val_str = str(raw_value).strip().lower()
    allowed_values = CATEGORICAL_VALUES[canonical_name]

    # 1. Exact case-insensitive match
    for allowed in allowed_values:
        if allowed.lower() == val_str:
            return allowed

    # 2. Synonyms and common typo dictionary match
    synonyms = _CATEGORICAL_SYNONYMS.get(canonical_name, {})
    if val_str in synonyms:
        return synonyms[val_str]

    for syn, canonical_val in synonyms.items():
        if syn in val_str or val_str in syn:
            return canonical_val

    # 3. Substring match against allowed values
    for allowed in allowed_values:
        if allowed.lower() in val_str or val_str in allowed.lower():
            return allowed

    # 4. Fuzzy string matching against allowed values + synonyms
    all_keys = list(synonyms.keys()) + [a.lower() for a in allowed_values]
    matches = difflib.get_close_matches(val_str, all_keys, n=1, cutoff=0.65)
    if matches:
        matched_key = matches[0]
        if matched_key in synonyms:
            return synonyms[matched_key]
        for allowed in allowed_values:
            if allowed.lower() == matched_key:
                return allowed

    return None


def validate_feature_value(feature_name: str, raw_value: Any) -> Tuple[bool, Any]:
    """
    Validate and normalize a value for a specific schema feature.
    - Numerical: parsed with multiplier and number-word awareness
    - Date: strict ISO-8601 string (YYYY-MM-DD)
    - Categorical: strictly validated against allowed dataset categories
    """
    is_valid_feat, canonical_name = validate_feature_name(feature_name)
    if not is_valid_feat or canonical_name is None:
        return False, None

    if raw_value is None:
        return False, None

    # Handle numerical features
    if canonical_name in NUMERICAL_FEATURES:
        if canonical_name == "Fuel_Efficiency":
            eff_val = convert_fuel_efficiency(raw_value)
            if eff_val is not None and eff_val >= 0:
                return True, eff_val
            return False, None

        if canonical_name == "Engine_Size":
            eng_val = convert_engine_size(raw_value)
            if eng_val is not None and eng_val >= 0:
                return True, eng_val
            return False, None

        parsed_num = parse_numerical_value(raw_value)
        if parsed_num is not None and parsed_num >= 0:
            return True, parsed_num
        return False, None

    # Handle date features
    if canonical_name in DATE_FEATURES:
        value = normalise_date_value(raw_value)
        if value:
            return True, value
        return False, None

    # Handle categorical features
    if canonical_name in CATEGORICAL_FEATURES:
        matched = fuzzy_match_categorical(canonical_name, raw_value)
        if matched:
            return True, matched
        return False, None

    return False, None


def extract_explicit_facts(
    user_message: str,
    current_question_feature: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    reference_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Deterministically extract facts from user message and active question context.
    Supports short contextual answers, natural language phrasing, spelling errors,
    multipliers (50k), unit conversions (L/100km, mpg, cc/Litre), dates,
    and isolates symptoms / replacement history from confirmed categorical state.
    """
    text = (user_message or "").strip()
    norm_text = normalize_typos_and_informal_tokens(text)
    lower = norm_text.lower()
    result: Dict[str, Any] = {
        "extracted_features": [],
        "unavailable_features": [],
        "declined_features": [],
        "unknown_features": [],
        "pending_features": [],
        "approximate_features": {},
        "symptoms": {},
        "replacement_history": {},
        "ambiguities": [],
        "contradictions": [],
        "unverified_unit_features": {},
        "temporal_candidates": {},
        "entity_observations": {},
        "raw_observations": [],
        "is_correction": False,
        "user_intent": "inform",
        "raw_user_message": text,
    }
    if not text:
        return result

    # Check for direct decline / uncertainty / pending / unavailable with active question
    if current_question_feature:
        is_valid_f, canon_f = validate_feature_name(current_question_feature)
        if is_valid_f and canon_f:
            # The message is targeted at another feature ONLY if it mentions another specific component/system
            # that is distinct from the active question's domain.
            feature_domain_keywords = {
                "Last_Service_Date": ("service", "serviced", "servicing", "when", "date", "last service", "oil change"),
                "Maintenance_History": ("maintenance", "maintain", "service", "serviced", "servicing", "history"),
                "Service_History": ("service", "services", "serviced", "visits", "times", "history"),
                "Battery_Status": ("battery", "batteries"),
                "Brake_Condition": ("brake", "brakes", "pads", "rotors"),
                "Tire_Condition": ("tire", "tires", "tyre", "tyres"),
                "Vehicle_Age": ("age", "old", "year", "bought", "purchased"),
                "Mileage": ("mileage", "distance", "km", "miles", "run", "driven", "covered"),
                "Odometer_Reading": ("odometer", "odo", "reading", "distance", "km", "miles"),
                "Reported_Issues": ("issue", "issues", "problem", "problems", "trouble", "faults"),
                "Accident_History": ("accident", "accidents", "crash", "crashes"),
                "Warranty_Expiry_Date": ("warranty", "expiry", "guarantee", "date"),
                "Fuel_Type": ("fuel", "gas", "petrol", "diesel", "electric"),
                "Transmission_Type": ("transmission", "gearbox", "gear", "clutch", "manual", "auto", "automatic"),
                "Engine_Size": ("engine", "motor", "cc", "displacement", "litre", "liter"),
                "Fuel_Efficiency": ("efficiency", "economy", "mileage", "kmpl", "mpg"),
                "Insurance_Premium": ("insurance", "premium"),
                "Owner_Type": ("owner", "ownership", "hand"),
                "Vehicle_Model": ("model", "car", "vehicle", "make"),
            }
            other_components = [
                comp for comp, kw_list in (
                    ("battery", ("battery", "batteru")),
                    ("brakes", ("brake", "brakes", "pads")),
                    ("tires", ("tire", "tires", "tyre", "tyres")),
                    ("engine", ("engine", "motor")),
                    ("service", ("service", "serviced", "oil change")),
                )
                if any(kw in lower for kw in kw_list)
            ]
            active_domain_kws = feature_domain_keywords.get(canon_f, (canon_f.lower().replace("_", " "),))
            user_mentions_active_domain = any(ak in lower for ak in active_domain_kws)
            user_mentions_other_component = any(
                comp not in canon_f.lower() and not any(comp in ak for ak in active_domain_kws)
                for comp in other_components
            )
            is_targeted_at_other = user_mentions_other_component and not user_mentions_active_domain
            if not is_targeted_at_other:
                if any(token in lower for token in _DECLINED_PATTERNS):
                    result["declined_features"].append(canon_f)
                    result["raw_observations"].append(f"User declined to provide {canon_f}: {text}")
                    result["user_intent"] = "decline"
                    return result
                if any(token in lower for token in _PENDING_PATTERNS):
                    result["pending_features"].append(canon_f)
                    result["raw_observations"].append(f"Feature {canon_f} is pending (user will check later): {text}")
                    result["user_intent"] = "pending"
                    return result
                if any(token in lower for token in _UNAVAILABLE_PATTERNS):
                    result["unavailable_features"].append(canon_f)
                    result["raw_observations"].append(f"User cannot remember {canon_f}: {text}")
                    result["user_intent"] = "decline"
                    return result
                if any(token in lower for token in _UNKNOWN_PATTERNS):
                    result["unknown_features"].append(canon_f)
                    result["raw_observations"].append(f"User does not know {canon_f}: {text}")
                    result["user_intent"] = "decline"
                    return result
            else:
                # Expressed uncertainty about an off-target specific historical detail
                result["raw_observations"].append(f"User expressed uncertainty regarding: {text}")
    else:
        # No active question: check for explicit targeted unavailability/unknown on specific features
        if any(token in lower for token in _UNAVAILABLE_PATTERNS):
            if any(w in lower for w in ("last service", "when i last serviced", "when it was serviced", "when i serviced", "service date")):
                if "Last_Service_Date" not in result["unavailable_features"]:
                    result["unavailable_features"].append("Last_Service_Date")
                    result["raw_observations"].append(f"User cannot remember Last_Service_Date: {text}")
        elif any(token in lower for token in _UNKNOWN_PATTERNS):
            if any(w in lower for w in ("mileage", "distance driven", "odometer")):
                if "Mileage" not in result["unknown_features"]:
                    result["unknown_features"].append("Mileage")
                    result["raw_observations"].append(f"User does not know Mileage: {text}")

    # Check for user correction intent across turns
    if any(p in lower for p in _CORRECTION_PATTERNS):
        result["is_correction"] = True

    # Antecedent resolution for pronouns
    antecedent = resolve_contextual_antecedents(norm_text, conversation_history, current_question_feature)
    if antecedent == "AMBIGUOUS":
        result["ambiguities"].append({
            "type": "ambiguous_pronoun_reference",
            "raw": text,
            "message": "Pronoun reference is ambiguous because multiple components were mentioned previously.",
        })
        result["raw_observations"].append(f"Ambiguous pronoun reference in: '{text}'.")
    elif antecedent:
        result["resolved_antecedent"] = antecedent

    # Dynamic temporal candidate extraction
    ref_dt = reference_date or datetime.now()
    temporal_cand = parse_temporal_expressions_and_intervals(norm_text, reference_date=ref_dt)
    if temporal_cand:
        result["temporal_candidates"]["temporal_expression"] = temporal_cand
        is_service_context = (
            current_question_feature in ("Maintenance_History", "Last_Service_Date", "Service_History")
            or any(k in lower for k in ("service", "serviced", "maintenance"))
            or antecedent == "service"
        )
        if is_service_context:
            result["temporal_candidates"]["Last_Service_Date"] = temporal_cand
            result["service_recency"] = temporal_cand["raw"]
            result["entity_observations"]["service_recency"] = temporal_cand["raw"]
            result["approximate_features"]["Last_Service_Date"] = temporal_cand["raw"]
            obs_msg = f"Vehicle service timing reported as approximately {temporal_cand['raw']} (candidate date ~{temporal_cand.get('approximate_date', 'unknown')})."
            if obs_msg not in result["raw_observations"]:
                result["raw_observations"].append(obs_msg)

    def add(name: str, value: Any) -> None:
        is_valid_f, canon_f = validate_feature_name(name)
        if is_valid_f and canon_f:
            is_valid_v, canon_v = validate_feature_value(canon_f, value)
            if is_valid_v and canon_v is not None:
                if not any(item["name"] == canon_f for item in result["extracted_features"]):
                    result["extracted_features"].append({"name": canon_f, "value": canon_v})
                    if any(p in lower for p in _APPROXIMATE_PATTERNS):
                        result["approximate_features"][canon_f] = text
                        result["raw_observations"].append(f"Approximate value provided for {canon_f}: {text}")

    # -------------------------------------------------------------------------
    # 0. Unsupported Attributes & Categories Safeguard
    # -------------------------------------------------------------------------
    if re.search(r"\b(?:4th|fourth)\s+(?:owner|hand)\b", lower):
        result["raw_observations"].append("Vehicle is 4th owner (exceeds 3rd owner model schema)")
    if re.search(r"\bhybrid\b", lower) and "Fuel_Type" not in [f["name"] for f in result["extracted_features"]]:
        result["raw_observations"].append("Powertrain is Hybrid (not in Petrol/Diesel/Electric schema)")
    if re.search(r"\b(?:cng|lpg)\b", lower):
        result["raw_observations"].append("Fuel is CNG/LPG (not in Petrol/Diesel/Electric schema)")
    if re.search(r"\bclean\s+title\b", lower):
        result["raw_observations"].append("Vehicle has clean title (does not prove 0 accidents)")

    # -------------------------------------------------------------------------
    # 1. Context-Aware Interpretation (Active Question Context)
    # -------------------------------------------------------------------------
    if current_question_feature:
        is_valid_curr, curr_canonical = validate_feature_name(current_question_feature)
        if is_valid_curr and curr_canonical:
            # Check numerical active question
            if curr_canonical in NUMERICAL_FEATURES:
                has_other_unit = (
                    ("problem" in lower or "issue" in lower) and curr_canonical != "Reported_Issues"
                ) or (
                    ("year" in lower or "yr" in lower) and curr_canonical != "Vehicle_Age"
                ) or (
                    ("service" in lower or "visit" in lower) and curr_canonical != "Service_History"
                ) or (
                    ("accident" in lower or "crash" in lower) and curr_canonical != "Accident_History"
                ) or (
                    "cc" in lower and curr_canonical != "Engine_Size"
                ) or (
                    ("mile" in lower or "km" in lower) and curr_canonical not in ("Mileage", "Odometer_Reading", "Fuel_Efficiency")
                )
                if not has_other_unit:
                    if curr_canonical == "Fuel_Efficiency":
                        eff_parsed = parse_fuel_efficiency_with_unit(text)
                        if eff_parsed:
                            if eff_parsed["is_generic_mpg"]:
                                amb = {
                                    "type": "generic_mpg_ambiguity",
                                    "feature": "Fuel_Efficiency",
                                    "value": eff_parsed["value"],
                                    "raw": text,
                                    "message": "Generic MPG stated without specifying US MPG or Imperial MPG. Clarification needed."
                                }
                                result["ambiguities"].append(amb)
                                result["unverified_unit_features"]["Fuel_Efficiency"] = eff_parsed
                                result["raw_observations"].append(
                                    f"User reported Fuel_Efficiency: {text} (Ambiguous between US MPG and Imperial MPG; canonical dataset unit unverified; preserving as observation)."
                                )
                            elif eff_parsed["unit"] != "unitless":
                                result["unverified_unit_features"]["Fuel_Efficiency"] = eff_parsed
                                result["raw_observations"].append(
                                    f"User reported Fuel_Efficiency: {text} (Canonical dataset unit is unverified; preserving as observation without model conversion)."
                                )
                            else:
                                add("Fuel_Efficiency", eff_parsed["value"])
                    elif curr_canonical in ("Mileage", "Odometer_Reading"):
                        unit_m = re.search(r"\b(miles|mile|km|kms|kilometres|kilometers)\b", lower)
                        if unit_m:
                            parsed_dist = parse_numerical_value(text)
                            if parsed_dist is not None:
                                add(curr_canonical, parsed_dist)
                                result["unverified_unit_features"][curr_canonical] = {
                                    "value": parsed_dist,
                                    "unit": unit_m.group(1),
                                    "raw": text,
                                }
                                result["raw_observations"].append(
                                    f"User reported {curr_canonical}: {parsed_dist} {unit_m.group(1)}."
                                )
                        else:
                            parsed_num = parse_numerical_value(text)
                            if parsed_num is not None:
                                add(curr_canonical, parsed_num)
                    elif curr_canonical == "Engine_Size":
                        eng_val = convert_engine_size(text)
                        if eng_val is not None:
                            add(curr_canonical, eng_val)
                    elif curr_canonical == "Vehicle_Age":
                        current_year = datetime.now().year
                        year_match = re.search(r"\b(19\d\d|20\d\d)\b", lower)
                        if year_match:
                            cal_year = int(year_match.group(1))
                            if 1980 <= cal_year <= current_year:
                                add("Vehicle_Age", current_year - cal_year)
                        else:
                            if not any(r in lower for r in ("replaced", "changed", "fitted", "serviced", "installed", "got new")) and not any(c in lower for c in ("battery", "brake", "brakes", "tire", "tires", "tyre", "tyres", "pads")):
                                parsed_num = parse_numerical_value(text)
                                if parsed_num is not None:
                                    add(curr_canonical, parsed_num)
                    else:
                        parsed_num = parse_numerical_value(text)
                        if parsed_num is not None:
                            add(curr_canonical, parsed_num)

            # Check categorical active question
            elif curr_canonical in CATEGORICAL_FEATURES:
                # Do not map symptoms to conditions even during active question
                is_symptom = False
                if curr_canonical == "Brake_Condition" and any(w in lower for w in ("squeak", "squeal", "grind", "soft", "spongy", "vibrat")):
                    is_symptom = True
                    result["raw_observations"].append(f"Brake symptom observed: {text}")
                elif curr_canonical == "Battery_Status" and any(w in lower for w in ("die", "dying", "dead", "crank", "jump")):
                    is_symptom = True
                    result["raw_observations"].append(f"Battery symptom observed: {text}")
                elif curr_canonical == "Tire_Condition" and any(w in lower for w in ("pressure", "puncture", "flat", "leak")):
                    is_symptom = True
                    result["raw_observations"].append(f"Tire symptom observed: {text}")

                # Do not map past replacement to "New" condition
                if any(w in lower for w in ("replaced", "changed")) and any(w in lower for w in ("ago", "last year", "last month", "months")):
                    is_symptom = True
                    result["raw_observations"].append(f"Replacement history mentioned: {text}")

                if not is_symptom:
                    conf_val, contra, obs_list, is_neg = evaluate_component_condition(
                        text, curr_canonical, (curr_canonical.lower().replace("_", " "),),
                        CATEGORICAL_VALUES.get(curr_canonical, []), _CATEGORICAL_SYNONYMS.get(curr_canonical, {})
                    )
                    if conf_val:
                        add(curr_canonical, conf_val)
                    elif curr_canonical == "Maintenance_History":
                        timing_val = parse_relative_service_timing(text)
                        if timing_val:
                            result["service_recency"] = timing_val
                            obs_msg = f"Vehicle service timing reported as approximately {timing_val}."
                            if obs_msg not in result["raw_observations"]:
                                result["raw_observations"].append(obs_msg)
                            result["ambiguities"].append({
                                "type": "active_question_timing_clarification",
                                "feature": "Maintenance_History",
                                "recency": timing_val,
                                "raw": text,
                                "message": f"User responded with service timing ('{timing_val}') rather than maintenance quality (Good, Average, Poor).",
                            })
                    if contra:
                        result["contradictions"].append(contra)
                    result["raw_observations"].extend(obs_list)

            # Check date active question
            elif curr_canonical in DATE_FEATURES:
                if curr_canonical == "Last_Service_Date" and any(p in lower for p in ("never serviced", "never been serviced", "not serviced yet", "never had a service", "no service done", "haven't serviced", "havent serviced")):
                    add("Service_History", 0)
                    result["unavailable_features"].append("Last_Service_Date")
                    result["raw_observations"].append("Vehicle has never been serviced; Last_Service_Date is unavailable and Service_History set to 0.")
                else:
                    date_val = normalise_date_value(text)
                    if date_val:
                        add(curr_canonical, date_val)
                    elif temporal_cand and temporal_cand.get("approximate_date"):
                        approx_iso = temporal_cand["approximate_date"]
                        add(curr_canonical, approx_iso)
                        result["approximate_features"][curr_canonical] = text
                        result["temporal_candidates"][curr_canonical] = temporal_cand
                    elif any(w in lower for w in ("month", "months", "year", "years", "ago", "last")):
                        result["raw_observations"].append(f"Relative date/duration for {curr_canonical}: {text}")

    # -------------------------------------------------------------------------
    # 2. Multi-Fact & Standalone Fact Patterns
    # -------------------------------------------------------------------------
    # If an antecedent component was resolved for a replacement expression
    if antecedent and antecedent in ("brakes", "battery", "tires"):
        if any(r in lower for r in ("replaced", "changed", "installed", "fitted", "got new", "bought new")) and any(t in lower for t in ("ago", "last year", "last month", "months", "year", "week", "yesterday", "recently")):
            result["replacement_history"][antecedent] = text
            obs_str = f"{antecedent.capitalize()} replacement history noted (resolved from context): {text}"
            if obs_str not in result["raw_observations"]:
                result["raw_observations"].append(obs_str)

    # If an antecedent component was resolved for a condition expression
    if antecedent and antecedent in ("battery", "brakes", "tires") and not any(r in lower for r in ("replaced", "changed", "installed")):
        target_feat = "Battery_Status" if antecedent == "battery" else ("Brake_Condition" if antecedent == "brakes" else "Tire_Condition")
        conf_val, contra, obs_list, is_neg = evaluate_component_condition(
            norm_text, target_feat, (antecedent,),
            CATEGORICAL_VALUES.get(target_feat, []), _CATEGORICAL_SYNONYMS.get(target_feat, {})
        )
        if conf_val:
            add(target_feat, conf_val)
        if contra:
            result["contradictions"].append(contra)
        result["raw_observations"].extend(obs_list)

    # Replacement History detection for components
    for comp_key, comp_canonical, comp_alias in [
        ("brakes", "Brake_Condition", ("brake", "brakes", "pads")),
        ("battery", "Battery_Status", ("battery", "batteru")),
        ("tires", "Tire_Condition", ("tire", "tires", "tyre", "tyres")),
    ]:
        repl_pattern = (
            rf"\b(?:replaced|changed|fitted|serviced|installed|got\s+new|bought\s+new|new)\s+(?:my\s+|the\s+)?(?:[\w\s]*?)(?:{'|'.join(comp_alias)})\s+(?:about\s+|around\s+)?(?:last\s+(?:month|year|week)|\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?|months?|weeks?)?\s*(?:ago)?\b|"
            rf"\b(?:my\s+|the\s+)?(?:{'|'.join(comp_alias)})\s+(?:was|were|got|been)?\s*(?:replaced|changed|fitted|serviced|installed)\s+(?:about\s+|around\s+)?(?:last\s+(?:month|year|week)|\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?|months?|weeks?)?\s*(?:ago)?\b|"
            rf"\b(?:installed|fitted|got|bought)\s+new\s+(?:{'|'.join(comp_alias)})\s+(?:last\s+(?:month|year|week)|(?:about\s+|around\s+)?(?:\d+|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?|months?)\s*(?:ago)?)\b"
        )
        m_repl = re.search(repl_pattern, lower)
        if not m_repl and any(k in lower for k in comp_alias):
            if any(r in lower for r in ("replaced", "changed", "installed", "fitted", "got new", "bought new", "new pads")) and any(t in lower for t in ("ago", "last year", "last month", "months", "year")):
                m_repl = True
        if m_repl:
            result["replacement_history"][comp_key] = text
            result["raw_observations"].append(f"{comp_key.capitalize()} replacement history noted: {text}")
            if temporal_cand:
                result["entity_observations"][f"{comp_key}_replacement_timing"] = temporal_cand["raw"]

    # Age: "2 years", "5 years old", "age is 2 years", "age 2", "vehicle is 5 years old", "bought in 2021"
    current_year = datetime.now().year
    cal_year_match = re.search(r"\b(?:bought|purchased|got)\s+(?:(?:the\s+)?(?:car|bike|vehicle|truck|suv|van|it)\s+)?(?:in\s+)?(19\d\d|20\d\d)\b|\b(19\d\d|20\d\d)\s+model\b", lower)
    if cal_year_match:
        yr = int(cal_year_match.group(1) or cal_year_match.group(2))
        if 1980 <= yr <= current_year:
            add("Vehicle_Age", current_year - yr)
    else:
        # Clause explicitly attributing age to the vehicle
        veh_age_match = re.search(
            r"\b(?:(?:my\s+)?(?:car|vehicle|bike|automobile|truck|suv|van|it)\s+is\s+|age\s*(?:is|=|:)\s*|vehicle\s+age\s*(?:is|=|:)\s*|car\s+age\s*(?:is|=|:)\s*)(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?)(?:\s+old)?\b|"
            r"\bage\s*(?:is|=|:)\s*(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\b",
            lower
        )
        if veh_age_match:
            val = next(g for g in veh_age_match.groups() if g is not None)
            parsed_age = parse_numerical_value(val)
            if parsed_age is not None:
                add("Vehicle_Age", parsed_age)
        else:
            # Standalone "... years old" pattern (e.g. "5 years old")
            comp_prefix = re.search(
                r"\b(?:battery|brake|brakes|tire|tires|tyre|tyres|pads)\s+(?:is|are|was|were)?\s*(?:about\s+|around\s+|almost\s+|nearly\s+|roughly\s+)?(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?)(?:\s+old)?\b",
                lower
            )
            if comp_prefix:
                obs_age = f"Component age mentioned: {text}"
                if obs_age not in result["raw_observations"]:
                    result["raw_observations"].append(obs_age)
            else:
                old_match = re.search(r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?)\s+old\b", lower)
                if old_match:
                    parsed_age = parse_numerical_value(old_match.group(1))
                    if parsed_age is not None:
                        add("Vehicle_Age", parsed_age)

    # Exception expression handling (e.g. "Everything is fine except the battery")
    m_except = re.search(r"\b(?:everything|all(?:\s+of\s+it)?|car|vehicle)\s+(?:is\s+|feels\s+|seems\s+)?(?:fine|good|okay|ok|in\s+order)\s*,?\s*(?:except|apart\s+from|but)\s+(?:the\s+|for\s+the\s+)?(\w+)\b", lower)
    if m_except:
        exc_comp = m_except.group(1).lower()
        result["raw_observations"].append(f"General vehicle condition reported as fine except for {exc_comp}.")
        add("Reported_Issues", 1)

    # Reported issues: "3 problems", "3 reported issues", "3 issues", "a couple of problems", "no problems"
    issue_match = re.search(r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:reported\s+)?(?:issues?|problems?|faults?)\b|\b(?:reported\s+)?(?:issues?|problems?)\s*(?:is|are|=|:)\s*(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\b", lower)
    if issue_match:
        val = next(g for g in issue_match.groups() if g is not None)
        parsed_issue = parse_numerical_value(val)
        if parsed_issue is not None:
            add("Reported_Issues", parsed_issue)
    elif any(p in lower for p in ("no issues", "no problems", "zero issues", "zero problems", "no complaints", "zero complaints", "everything is fine", "everything is good", "everything fine", "everything good", "everything is ok", "all is good", "all is fine")) and not m_except:
        add("Reported_Issues", 0)

    # Service history: "4 services", "serviced 3 times", "4 past services", "3 service visits", "never serviced"
    if "never serviced" in lower or "no services" in lower or "zero services" in lower:
        add("Service_History", 0)
    else:
        service_match = re.search(
            r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:past\s+)?services?\b|"
            r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:past\s+)?service\s*(?:visits?|appointments?)\b|"
            r"\b(?:serviced|servied|service)\s+(?:it|the\s+vehicle)?\s*(?:for\s+)?(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*times\b|"
            r"\b(?:went\s+to\s+(?:the\s+)?(?:dealer|dealership|service\s+center|mechanic|garage|shop))\s+(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*times\b|"
            r"\bservice\s*history\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\b",
            lower
        )
        if service_match:
            val = next(g for g in service_match.groups() if g is not None)
            parsed_svc = parse_numerical_value(val)
            if parsed_svc is not None:
                add("Service_History", parsed_svc)

    # Accident history: "0 accidents", "no accidents", "never had an accident", "2 accidents"
    accident_match = re.search(
        r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:past\s+)?(?:accidents?|crashes?)\b|"
        r"\bno\s+accidents?\b|\bnever\s+(?:had\s+an\s+|been\s+in\s+an\s+)?accidents?\b|"
        r"\baccident\s*history\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\b|"
        r"\b(?:crashed|had\s+a\s+crash)\s+(\d+|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)(?:\s+times?)?\b|"
        r"\bcrashed\s+(once|twice)\b",
        lower
    )
    if accident_match:
        if "no accidents" in lower or "never" in lower:
            add("Accident_History", 0)
        else:
            val = next(g for g in accident_match.groups() if g is not None)
            if val == "once":
                add("Accident_History", 1)
            elif val == "twice":
                add("Accident_History", 2)
            else:
                parsed_acc = parse_numerical_value(val)
                if parsed_acc is not None:
                    add("Accident_History", parsed_acc)

    # Fuel Efficiency: check units and ambiguity without assuming canonical dataset units
    eff_parsed = parse_fuel_efficiency_with_unit(lower)
    has_eff_mention = bool(re.search(r"\b(?:fuel\s+(?:efficiency|economy)|consumption|km/?l|kmpl|l/100km|l/100\s*km|mpg|miles?\s*per\s*gallon)\b", lower))
    if eff_parsed and has_eff_mention:
        if eff_parsed["is_generic_mpg"]:
            amb = {
                "type": "generic_mpg_ambiguity",
                "feature": "Fuel_Efficiency",
                "value": eff_parsed["value"],
                "raw": text,
                "message": "Generic MPG stated without specifying US MPG or Imperial MPG. Clarification needed."
            }
            result["ambiguities"].append(amb)
            result["unverified_unit_features"]["Fuel_Efficiency"] = eff_parsed
            result["raw_observations"].append(
                f"User reported Fuel_Efficiency: {text} (Ambiguous between US MPG and Imperial MPG; canonical dataset unit unverified; preserving as observation)."
            )
        elif eff_parsed["unit"] != "unitless":
            result["unverified_unit_features"]["Fuel_Efficiency"] = eff_parsed
            result["raw_observations"].append(
                f"User reported Fuel_Efficiency: {text} (Canonical dataset unit is unverified; preserving as observation without model conversion)."
            )
        else:
            add("Fuel_Efficiency", eff_parsed["value"])

    # Mileage & Odometer (Cumulative Distance Only)
    mileage_unit_match = re.search(r"\b(?:mileage\s*(?:is|=|:)?\s*|(?:has\s+run|has\s+covered|has\s+done|has\s+clocked|driven|run|covered|done|clocked|travelled|traveled)\s+(?:about\s+|around\s+)?)([\d,]+(?:\.\d+)?\s*(?:k|thousand|lakh|lac|m|million)?)\s*(miles|mile|km|kms|kilometres|kilometers)\b", lower)
    if mileage_unit_match and not has_eff_mention:
        parsed_dist = parse_numerical_value(mileage_unit_match.group(1))
        stated_unit = mileage_unit_match.group(2)
        if parsed_dist is not None:
            result["unverified_unit_features"]["Mileage"] = {
                "value": parsed_dist,
                "unit": stated_unit,
                "raw": text,
            }
            result["raw_observations"].append(
                f"User reported Mileage: {parsed_dist} {stated_unit} (Canonical dataset distance unit is unverified; preserving as observation without model conversion)."
            )
    else:
        mileage_nounit_match = re.search(r"\b(?:mileage\s*(?:is|=|:)?\s*|(?:has\s+run|has\s+covered|has\s+done|has\s+clocked|driven|run|covered|done|clocked|travelled|traveled)\s+(?:about\s+|around\s+)?)([\d,]+(?:\.\d+)?\s*(?:k|thousand|lakh|lac|m|million)?)\b", lower)
        if mileage_nounit_match and not has_eff_mention and not re.search(r"\b(?:miles?|kms?|kilometres?|kilometers?|km/?l|kmpl|mpg|l/100)\b", lower):
            parsed_mileage = parse_numerical_value(mileage_nounit_match.group(1))
            if parsed_mileage is not None:
                add("Mileage", parsed_mileage)

    # Check Odometer explicitly
    odometer_unit_match = re.search(r"\b(?:odometer|odo)(?:\s+reading)?\s*(?:is|=|:|shows|reads|at)?\s*([\d,]+(?:\.\d+)?\s*(?:k|thousand|lakh|lac|m|million)?)\s*(miles|mile|km|kms|kilometres|kilometers)\b", lower)
    if odometer_unit_match:
        parsed_dist = parse_numerical_value(odometer_unit_match.group(1))
        stated_unit = odometer_unit_match.group(2)
        if parsed_dist is not None:
            result["unverified_unit_features"]["Odometer_Reading"] = {
                "value": parsed_dist,
                "unit": stated_unit,
                "raw": text,
            }
            result["raw_observations"].append(
                f"User reported Odometer_Reading: {parsed_dist} {stated_unit} (Canonical dataset distance unit is unverified; preserving as observation without model conversion)."
            )
    else:
        odometer_nounit_match = re.search(r"\b(?:odometer|odo)(?:\s+reading)?\s*(?:is|=|:|shows|reads|at)?\s*([\d,]+(?:\.\d+)?\s*(?:k|thousand|lakh|lac|m|million)?)\b", lower)
        if odometer_nounit_match and not any(u in lower for u in ("kmpl", "mpg", "l/100")) and not re.search(r"\b(?:miles?|kms?|kilometres?|kilometers?)\b", lower):
            parsed_odo = parse_numerical_value(odometer_nounit_match.group(1))
            if parsed_odo is not None:
                add("Odometer_Reading", parsed_odo)

    # Engine size: "1500 cc", "1.5L", "1.5 litre", "2.0 liter"
    engine_match = re.search(r"\b(?:engine\s*(?:size)?\s*(?:is|=|:)?\s*)?(\d+(?:\.\d+)?\s*(?:cc|cubic\s*centimet(?:er|re)s?|l|litre|liter|litres|liters))\b", lower)
    if engine_match:
        eng_val = convert_engine_size(engine_match.group(1))
        if eng_val is not None:
            add("Engine_Size", eng_val)

    # Insurance premium: "insurance is $15,000", "paying 18k premium", "insurance cost 22000"
    insurance_match = re.search(r"\b(?:insurance(?:\s+premium)?|premium)\s*(?:is|=|:)?\s*[$€£₹]?\s*([\d,]+(?:\.\d+)?\s*(?:k|thousand|lakh|lac|m|million)?)\b", lower)
    if insurance_match:
        ins_val = parse_numerical_value(insurance_match.group(1))
        if ins_val is not None:
            add("Insurance_Premium", ins_val)

    # Owner Type: "first owner", "1st owner", "second hand", "2 owners", "third owner"
    if not re.search(r"\b(?:4th|fourth)\b", lower):
        owner_match = re.search(r"\b(first|1st|single|one|second|2nd|two|third|3rd|three)\s+(?:owner|hand|owners)\b|\bbought\s+(?:it\s+)?(?:brand\s+)?new\b", lower)
        if owner_match:
            cat_val = fuzzy_match_categorical("Owner_Type", lower)
            if cat_val:
                add("Owner_Type", cat_val)

    # Transmission Type: explicit statement e.g. "transmission is automatic", "manual gearbox", "it's automatic", "stick shift"
    trans_match = re.search(
        r"\b(?:transmission(?:\s+type)?|gearbox)\s*(?:is|=|:)?\s*(automatic|manual|auto|stick\s*shift|amt|cvt)\b|"
        r"\b(?:it\s*is\s*an?|it'?s\s*an?|car\s+is\s*an?)\s*(automatic|manual)\b|"
        r"\b(?:standard\s+)?(manual|automatic|stick\s*shift)\s+(?:transmission|gearbox)\b|"
        r"\b(stick\s*shift)\b",
        lower
    )
    if trans_match:
        cat_val = fuzzy_match_categorical("Transmission_Type", next(g for g in trans_match.groups() if g is not None))
        if cat_val:
            add("Transmission_Type", cat_val)

    # Fuel Type: explicit statement e.g. "fuel is petrol", "runs on diesel", "it's an electric car"
    fuel_match = re.search(r"\b(?:fuel(?:\s+type)?)\s*(?:is|=|:)?\s*(petrol|gasoline|diesel|electric)\b|\b(?:runs\s+on|takes)\s+(petrol|gasoline|diesel|electric)\b|\b(?:it\s*is\s*an?|it'?s\s*an?)\s*(electric|diesel|petrol)\b", lower)
    if fuel_match:
        cat_val = fuzzy_match_categorical("Fuel_Type", next(g for g in fuel_match.groups() if g is not None))
        if cat_val:
            add("Fuel_Type", cat_val)

    # Vehicle Model: explicit statement e.g. "model is car", "vehicle type is suv", "it's a motorcycle", "body type is sedan", "pickup truck"
    model_match = re.search(
        r"\b(?:vehicle\s+(?:model|type)|model|body\s+type)\s*(?:is|=|:)?\s*(car|sedan|hatchback|coupe|wagon|suv|crossover|truck|pickup(?:\s+truck)?|van|minivan|motorcycle|bike|scooter|bus)\b|"
        r"\b(?:it\s*is\s*an?|it'?s\s*an?|type\s*is\s*an?)\s*(car|sedan|hatchback|coupe|wagon|suv|crossover|truck|pickup(?:\s+truck)?|van|minivan|motorcycle|bike|scooter|bus)\b|"
        r"^(?:a\s+|an\s+)?(suv|pickup(?:\s+truck)?|sedan|hatchback|coupe|minivan|crossover|motorcycle|truck)\s*$",
        lower
    )
    if model_match:
        cat_val = fuzzy_match_categorical("Vehicle_Model", next(g for g in model_match.groups() if g is not None))
        if cat_val:
            add("Vehicle_Model", cat_val)

    # -------------------------------------------------------------------------
    # 3. Component Conditions vs. Symptoms vs. Replacement History
    # -------------------------------------------------------------------------
    # Subjective / Vague condition expressions (e.g. "battery is kind of old", "brakes are somewhat worn", "a bit old")
    vague_cond_match = re.search(r"\b(?:kind\s+of|sort\s+of|somewhat|a\s+bit|a\s+little)\s+(?:old|worn|tired|aged)\b", lower)
    if vague_cond_match:
        for comp_name, comp_feat, comp_kws in (
            ("battery", "Battery_Status", ("battery", "batteru", "batery")),
            ("brakes", "Brake_Condition", ("brake", "brakes", "pads")),
            ("tires", "Tire_Condition", ("tire", "tires", "tyre", "tyres")),
        ):
            is_relevant = any(k in lower for k in comp_kws) or (current_question_feature == comp_feat)
            if is_relevant:
                amb = {
                    "type": "subjective_condition_ambiguity",
                    "feature": comp_feat,
                    "raw": text,
                    "message": f"Subjective description ('{vague_cond_match.group(0)}') requires clarification regarding exact condition for {comp_feat}.",
                }
                if not any(a.get("feature") == comp_feat and a.get("type") == "subjective_condition_ambiguity" for a in result["ambiguities"]):
                    result["ambiguities"].append(amb)
                    obs = f"Subjective condition '{vague_cond_match.group(0)}' for {comp_feat}: clarification needed."
                    if obs not in result["raw_observations"]:
                        result["raw_observations"].append(obs)

    # Brakes
    if any(k in lower for k in ("brake", "brakes", "brke", "breaks", "break", "pads")):
        is_symptom = any(s in lower for s in ("squeak", "squeal", "grind", "soft", "spongy", "vibrat", "shudder", "sound", "noise"))
        is_replacement = "brakes" in result["replacement_history"]
        is_exception_target = bool(m_except and any(alias in m_except.group(1).lower() for alias in ("brake", "brakes", "pads")))
        if is_symptom:
            result["symptoms"]["brakes"] = text
            result["raw_observations"].append(f"Brake symptom noted: {text}")
        if not is_replacement and not is_exception_target:
            conf_val, contra, obs_list, is_neg = evaluate_component_condition(
                text, "Brake_Condition", ("brake", "brakes", "brke", "breaks", "break", "pads"),
                CATEGORICAL_VALUES["Brake_Condition"], _CATEGORICAL_SYNONYMS["Brake_Condition"]
            )
            if conf_val:
                add("Brake_Condition", conf_val)
            if contra:
                result["contradictions"].append(contra)
            result["raw_observations"].extend(obs_list)

    # Battery
    if any(k in lower for k in ("battery", "batteru", "batery")):
        is_symptom = any(s in lower for s in ("die", "dying", "dead", "won't start", "wont start", "jump", "crank", "draining", "discharge"))
        is_replacement = "battery" in result["replacement_history"]
        is_exception_target = bool(m_except and any(alias in m_except.group(1).lower() for alias in ("battery", "batteru", "batery")))
        if is_symptom:
            result["symptoms"]["battery"] = text
            result["raw_observations"].append(f"Battery symptom noted: {text}")
        if not is_replacement and not is_exception_target:
            conf_val, contra, obs_list, is_neg = evaluate_component_condition(
                text, "Battery_Status", ("battery", "batteru", "batery"),
                CATEGORICAL_VALUES["Battery_Status"], _CATEGORICAL_SYNONYMS["Battery_Status"]
            )
            if conf_val:
                add("Battery_Status", conf_val)
            if contra:
                result["contradictions"].append(contra)
            result["raw_observations"].extend(obs_list)

    # Tires
    if any(k in lower for k in ("tire", "tires", "tyre", "tyres")):
        is_symptom = any(s in lower for s in ("pressure", "puncture", "flat", "leak", "vibrat", "noise"))
        is_replacement = "tires" in result["replacement_history"]
        is_exception_target = bool(m_except and any(alias in m_except.group(1).lower() for alias in ("tire", "tires", "tyre", "tyres")))
        if is_symptom:
            result["symptoms"]["tires"] = text
            result["raw_observations"].append(f"Tire symptom noted: {text}")
        if not is_replacement and not is_exception_target:
            conf_val, contra, obs_list, is_neg = evaluate_component_condition(
                text, "Tire_Condition", ("tire", "tires", "tyre", "tyres"),
                CATEGORICAL_VALUES["Tire_Condition"], _CATEGORICAL_SYNONYMS["Tire_Condition"]
            )
            if conf_val:
                add("Tire_Condition", conf_val)
            if contra:
                result["contradictions"].append(contra)
            result["raw_observations"].extend(obs_list)

    # Engine
    if any(k in lower for k in ("engine", "motor")):
        is_symptom = any(s in lower for s in (
            "struggle", "start", "starting", "vibrat", "shake", "shaking", "smoke",
            "overheat", "misfire", "stall", "noise", "sound", "temperature", "temp",
            "rises", "runs hot", "heating", "hesitat", "coolant"
        ))
        is_negated = any(p in lower for p in (
            "not making", "no unusual", "no strange", "running fine", "running well",
            "engine is fine", "engine fine", "motor fine", "not overheat", "no overheat",
            "doesn't overheat", "does not overheat", "never overheat", "no heating issue",
            "no issues with heating", "no temperature issue"
        ))
        if is_symptom and not is_negated:
            result["symptoms"]["engine"] = text
            result["raw_observations"].append(f"Engine symptom noted: {text}")

    # Steering
    if any(k in lower for k in ("steering", "wheel", "align")):
        is_symptom = any(s in lower for s in ("heavy", "stiff", "loose", "pull", "pulling", "vibrat", "shak"))
        if is_symptom:
            result["symptoms"]["steering"] = text
            result["raw_observations"].append(f"Steering symptom noted: {text}")

    # Exhaust
    if any(k in lower for k in ("exhaust", "tailpipe", "muffler")):
        is_symptom = any(s in lower for s in ("smoke", "loud", "fume", "leak", "smell", "black smoke", "white smoke"))
        if is_symptom:
            result["symptoms"]["exhaust"] = text
            result["raw_observations"].append(f"Exhaust symptom noted: {text}")

    # General drivability / vehicle trouble
    if any(p in lower for p in ("giving trouble", "giving me trouble", "giving problems", "doesn't feel right", "doesnt feel right", "feels weird", "struggling")):
        if "general" not in result["symptoms"]:
            result["symptoms"]["general"] = text
            result["raw_observations"].append(f"General vehicle drivability symptom noted: {text}")

    # Maintenance history: "maintenance is average", "maintainance is avrage", "maintenence average", "maintenance avg"
    if any(k in lower for k in ("maintenance", "maintainance", "maintenence", "maintanence")):
        conf_val, contra, obs_list, is_neg = evaluate_component_condition(
            text, "Maintenance_History", ("maintenance", "maintainance", "maintenence", "maintanence"),
            CATEGORICAL_VALUES["Maintenance_History"], _CATEGORICAL_SYNONYMS["Maintenance_History"]
        )
        if conf_val:
            add("Maintenance_History", conf_val)
        if contra:
            result["contradictions"].append(contra)
        result["raw_observations"].extend(obs_list)

    # Date extraction: exact dates and approximate relative service dates
    date_val = normalise_date_value(text)
    if date_val:
        if any(k in lower for k in ("last service", "last serviced", "serviced on", "serviced", "service was", "service date")):
            add("Last_Service_Date", date_val)
        elif any(k in lower for k in ("warranty", "warranty expiry")):
            add("Warranty_Expiry_Date", date_val)
    elif temporal_cand and temporal_cand.get("approximate_date"):
        if any(k in lower for k in ("last service", "last serviced", "service date")):
            add("Last_Service_Date", temporal_cand["approximate_date"])
            result["approximate_features"]["Last_Service_Date"] = text
            result["temporal_candidates"]["Last_Service_Date"] = temporal_cand
        elif not current_question_feature and any(k in lower for k in ("serviced on", "service was")):
            add("Last_Service_Date", temporal_cand["approximate_date"])
            result["approximate_features"]["Last_Service_Date"] = text
            result["temporal_candidates"]["Last_Service_Date"] = temporal_cand
        elif any(k in lower for k in ("warranty", "warranty expiry")):
            add("Warranty_Expiry_Date", temporal_cand["approximate_date"])
            result["approximate_features"]["Warranty_Expiry_Date"] = text
            result["temporal_candidates"]["Warranty_Expiry_Date"] = temporal_cand
    elif any(k in lower for k in ("service", "warranty", "maintenance")) and any(w in lower for w in ("month", "months", "year", "years", "ago", "last", "recently")):
        rel_timing = parse_relative_service_timing(text)
        if rel_timing and "service_recency" not in result:
            result["service_recency"] = rel_timing
            obs_msg = f"Vehicle service timing reported as approximately {rel_timing}."
            if obs_msg not in result["raw_observations"]:
                result["raw_observations"].append(obs_msg)
        else:
            result["raw_observations"].append(f"Relative date/duration mentioned: {text}")

    # Standalone ambiguous numbers (e.g. "50000" or "45k" with no feature keyword and no active question)
    if not current_question_feature and not result["extracted_features"] and not result["symptoms"] and not result["replacement_history"]:
        bare_num_match = re.search(r"^\s*(?:about\s+|approx\s+|around\s+)?(\d{1,3}(?:,\d{3})+|\d+)\s*(?:k)?\s*$", lower)
        if bare_num_match and not any(unit in lower for unit in ("km", "mile", "kmpl", "mpg", "cc", "l", "year", "yr", "accident", "issue", "service")) and not any(feat in lower for feat in ("mileage", "odometer", "reading", "premium", "insurance", "age", "history", "cost")):
            val = parse_numerical_value(text)
            if val is not None:
                amb = {"type": "ambiguous_number", "value": val, "raw": text}
                result["ambiguities"].append(amb)
                result["raw_observations"].append(
                    f"Ambiguous numerical value: '{text}'. Clarification needed whether this represents Mileage, Odometer Reading, or other metric."
                )

    # Intent detection
    if "brake" in lower:
        result["user_intent"] = "brake"
    elif "battery" in lower:
        result["user_intent"] = "battery"
    elif "tire" in lower or "tyre" in lower:
        result["user_intent"] = "tire"
    elif any(p in lower for p in ("won't start", "wont start", "doesn't start", "doesnt start", "starting")):
        result["user_intent"] = "starting"
    elif ("overheat" in lower or any(w in lower for w in ("temperature", "temp rises", "runs hot", "heating"))) and not any(p in lower for p in ("not overheat", "no overheat", "doesn't overheat", "does not overheat", "never overheat", "no heating issue", "no issues with heating")):
        result["user_intent"] = "overheating"
    elif any(k in lower for k in ("engine", "motor")):
        result["user_intent"] = "engine"
    elif "accident" in lower or "crash" in lower:
        result["user_intent"] = "accident"
    elif "service" in lower or "maintenance" in lower:
        result["user_intent"] = "maintenance_history"

    if not result["extracted_features"] and not result["unavailable_features"] and not result["symptoms"] and not result["replacement_history"] and not result["ambiguities"]:
        if text not in result["raw_observations"]:
            result["raw_observations"].append(text)

    return result


def constrain_extraction_to_explicit_facts(
    extraction: Dict[str, Any],
    user_message: str,
    current_question_feature: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    reference_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Intersect and merge InputAgent LLM extraction with deterministic extraction.
    Guarantees strict schema adherence while supporting natural language understanding.
    """
    evidenced = extract_explicit_facts(
        user_message, current_question_feature, conversation_history, reference_date=reference_date
    )

    # Process InputAgent extractions (accepts list of dicts or name:value dict)
    llm_extracted = extraction.get("extracted_features", []) if isinstance(extraction, dict) else []
    items_to_add = []
    if isinstance(llm_extracted, list):
        for item in llm_extracted:
            if isinstance(item, dict):
                items_to_add.append((item.get("name", ""), item.get("value")))
    elif isinstance(llm_extracted, dict):
        for name, val in llm_extracted.items():
            items_to_add.append((name, val))

    for name, val in items_to_add:
        is_valid_f, canonical_f = validate_feature_name(name)
        if not is_valid_f or canonical_f is None:
            continue

        is_valid_v, canonical_v = validate_feature_value(canonical_f, val)
        if not is_valid_v or canonical_v is None:
            continue

        # Add if not already present
        if not any(f["name"] == canonical_f for f in evidenced["extracted_features"]):
            evidenced["extracted_features"].append({"name": canonical_f, "value": canonical_v})

    # Process unavailable / declined / unknown / pending features
    if isinstance(extraction, dict):
        for key in ("unavailable_features", "declined_features", "unknown_features", "pending_features"):
            llm_list = extraction.get(key, [])
            if isinstance(llm_list, list):
                for name in llm_list:
                    is_valid_f, canonical_f = validate_feature_name(name)
                    if is_valid_f and canonical_f and canonical_f not in evidenced.get(key, []):
                        evidenced.setdefault(key, []).append(canonical_f)

        # Merge symptoms and replacement history
        if isinstance(extraction.get("symptoms"), dict):
            evidenced.setdefault("symptoms", {}).update(extraction["symptoms"])
        if isinstance(extraction.get("replacement_history"), dict):
            evidenced.setdefault("replacement_history", {}).update(extraction["replacement_history"])

        # Merge ambiguities
        if isinstance(extraction.get("ambiguities"), list):
            evidenced.setdefault("ambiguities", []).extend(extraction["ambiguities"])

        # Merge contradictions
        if isinstance(extraction.get("contradictions"), list):
            evidenced.setdefault("contradictions", []).extend(extraction["contradictions"])

        # Merge unverified unit features
        if isinstance(extraction.get("unverified_unit_features"), dict):
            evidenced.setdefault("unverified_unit_features", {}).update(extraction["unverified_unit_features"])

        # Merge approximate features
        if isinstance(extraction.get("approximate_features"), dict):
            evidenced.setdefault("approximate_features", {}).update(extraction["approximate_features"])

        # Merge temporal candidates
        if isinstance(extraction.get("temporal_candidates"), dict):
            evidenced.setdefault("temporal_candidates", {}).update(extraction["temporal_candidates"])

        # Merge entity observations
        if isinstance(extraction.get("entity_observations"), dict):
            evidenced.setdefault("entity_observations", {}).update(extraction["entity_observations"])

        # Merge correction flag
        if extraction.get("is_correction"):
            evidenced["is_correction"] = True

        # User intent
        if extraction.get("user_intent"):
            evidenced["user_intent"] = extraction["user_intent"]

    # Preserve raw observations
    llm_obs = extraction.get("raw_observations", []) if isinstance(extraction, dict) else []
    if isinstance(llm_obs, list):
        for obs in llm_obs:
            if obs and isinstance(obs, str) and obs not in evidenced["raw_observations"]:
                evidenced["raw_observations"].append(obs)

    return evidenced


def apply_structured_extraction(
    state: AgentState,
    extraction: Dict[str, Any],
    raw_user_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate LLM extraction, detect contradictions and ambiguities,
    route symptoms and component replacement history, and update AgentState deterministically.
    """
    state_changed = False
    updated_features: List[str] = []
    declined_features: List[str] = []
    rejected_features: List[str] = []
    contradictions: List[Dict[str, Any]] = []
    ambiguities: List[Dict[str, Any]] = []

    known_features = state.get_known_features()
    is_correction = bool(extraction.get("is_correction", False))

    # 1. Process explicit feature extractions (accepts list of dicts or name:value dict)
    raw_extracted = extraction.get("extracted_features", [])
    items_to_process = []
    if isinstance(raw_extracted, list):
        for item in raw_extracted:
            if isinstance(item, dict):
                items_to_process.append((item.get("name", ""), item.get("value")))
    elif isinstance(raw_extracted, dict):
        for name, val in raw_extracted.items():
            items_to_process.append((name, val))

    for name, val in items_to_process:
        is_valid_feat, canonical_name = validate_feature_name(name)
        if not is_valid_feat or canonical_name is None:
            logger.warning(f"[FEATURE_TOOL] Rejected non-schema feature: '{name}'")
            rejected_features.append(name)
            continue

        is_valid_val, canonical_val = validate_feature_value(canonical_name, val)
        if not is_valid_val or canonical_val is None:
            logger.warning(f"[FEATURE_TOOL] Rejected invalid value for '{canonical_name}': {val}")
            rejected_features.append(f"{canonical_name}:{val}")
            continue

        # Contradiction Detection against existing confirmed facts
        if canonical_name in known_features:
            existing_val = known_features[canonical_name]
            if existing_val != canonical_val:
                if is_correction:
                    logger.info(f"[STATE] Correction applied for {canonical_name}: {existing_val} -> {canonical_val}")
                    state.raw_observations.append(f"Correction applied for {canonical_name}: updated from {existing_val} to {canonical_val}")
                else:
                    logger.warning(f"[STATE] Contradiction detected for {canonical_name}: confirmed {existing_val} vs new {canonical_val}")
                    contra = {
                        "feature": canonical_name,
                        "existing_value": existing_val,
                        "conflicting_value": canonical_val,
                    }
                    contradictions.append(contra)
                    state.raw_user_inputs.setdefault("contradictions", []).append(contra)
                    state.raw_observations.append(
                        f"Contradiction detected for {canonical_name}: previously confirmed as {existing_val}, but user stated {canonical_val}. Clarification needed."
                    )
                    rejected_features.append(f"{canonical_name}:contradiction_with_{existing_val}")
                    continue

        # Update feature in state (also automatically un-declines if previously declined)
        changed = state.update_feature(canonical_name, canonical_val, status="known")
        if changed or state.feature_state[canonical_name].get("status") == "known":
            state_changed = True
            if canonical_name not in updated_features:
                updated_features.append(canonical_name)
            # Ensure purged from all uncertainty tracking
            for u_key in ("unknown_features", "unavailable_features", "pending_features", "declined_features"):
                u_list = state.raw_user_inputs.get(u_key)
                if isinstance(u_list, list) and canonical_name in u_list:
                    state.raw_user_inputs[u_key] = [x for x in u_list if x != canonical_name]
            state.declined_features.discard(canonical_name)
            logger.info(f"[STATE] Confirmed {canonical_name} = {canonical_val}")

    # 2. Process declined / unavailable / unknown / pending features
    # (A) Explicit declined
    raw_declined = extraction.get("declined_features", [])
    if isinstance(raw_declined, list):
        for name in raw_declined:
            is_valid_feat, canonical_name = validate_feature_name(name)
            if is_valid_feat and canonical_name:
                state.mark_declined(canonical_name)
                state.raw_user_inputs.setdefault("declined_features", []).append(canonical_name)
                if canonical_name not in declined_features:
                    declined_features.append(canonical_name)
                state_changed = True
                logger.info(f"[STATE] User explicitly declined {canonical_name} (will not ask again)")

    # (B) Unavailable ("can't remember") - independently decoupled from declined!
    raw_unavail = extraction.get("unavailable_features", [])
    if isinstance(raw_unavail, list):
        for name in raw_unavail:
            is_valid_feat, canonical_name = validate_feature_name(name)
            if is_valid_feat and canonical_name:
                state.mark_unavailable(canonical_name)
                state.raw_user_inputs.setdefault("unavailable_features", []).append(canonical_name)
                state_changed = True
                logger.info(f"[STATE] Marked {canonical_name} as unavailable (user cannot remember)")

    # (C) Unknown ("don't know") - NOT marked as declined!
    raw_unknown = extraction.get("unknown_features", [])
    if isinstance(raw_unknown, list):
        for name in raw_unknown:
            is_valid_feat, canonical_name = validate_feature_name(name)
            if is_valid_feat and canonical_name:
                state.feature_state[canonical_name]["status"] = "unknown"
                state.feature_state[canonical_name]["value"] = None
                state.raw_user_inputs.setdefault("unknown_features", []).append(canonical_name)
                state.asked_features.add(canonical_name)
                if state.active_ambiguity and state.active_ambiguity.get("feature") == canonical_name:
                    state.active_ambiguity = None
                if state.current_question_feature == canonical_name:
                    state.current_question_feature = None
                state_changed = True
                logger.info(f"[STATE] Marked {canonical_name} as unknown (user does not know)")

    # (D) Pending ("I'll check and tell you")
    raw_pending = extraction.get("pending_features", [])
    if isinstance(raw_pending, list):
        for name in raw_pending:
            is_valid_feat, canonical_name = validate_feature_name(name)
            if is_valid_feat and canonical_name:
                state.feature_state[canonical_name]["status"] = "unknown"
                state.feature_state[canonical_name]["value"] = None
                state.raw_user_inputs.setdefault("pending_features", []).append(canonical_name)
                state.asked_features.add(canonical_name)
                if state.active_ambiguity and state.active_ambiguity.get("feature") == canonical_name:
                    state.active_ambiguity = None
                if state.current_question_feature == canonical_name:
                    state.current_question_feature = None
                state_changed = True
                logger.info(f"[STATE] Marked {canonical_name} as pending (user will check later)")

    # Store intra-turn contradictions
    raw_contra = extraction.get("contradictions", [])
    if isinstance(raw_contra, list):
        for c in raw_contra:
            contradictions.append(c)
            state.raw_user_inputs.setdefault("contradictions", []).append(c)

    # Store unverified unit features
    unverified_units = extraction.get("unverified_unit_features", {})
    if isinstance(unverified_units, dict):
        state.raw_user_inputs.setdefault("unverified_unit_features", {}).update(unverified_units)

    # 3. Store symptoms
    symptoms_dict = extraction.get("symptoms", {})
    if isinstance(symptoms_dict, dict):
        for comp, symp in symptoms_dict.items():
            if symp:
                state.symptoms[comp] = symp
                obs_text = f"{comp.capitalize()} symptom: {symp}"
                if obs_text not in state.raw_observations:
                    state.raw_observations.append(obs_text)

    # 4. Store component replacement history
    repl_dict = extraction.get("replacement_history", {})
    if isinstance(repl_dict, dict):
        for comp, hist in repl_dict.items():
            if hist:
                state.replacement_history[comp] = hist
                state.raw_user_inputs.setdefault("replacement_history", {})[comp] = hist
                obs_text = f"{comp.capitalize()} replacement history: {hist}"
                if obs_text not in state.raw_observations:
                    state.raw_observations.append(obs_text)

    # 4b. Store service recency observation & entity revision
    recency = extraction.get("service_recency")
    if recency:
        prev_recency = state.raw_user_inputs.get("service_recency")
        if is_correction and prev_recency and prev_recency != recency:
            corr_obs = f"Correction applied: service recency updated from {prev_recency} to {recency}."
            state.raw_observations.append(corr_obs)
            logger.info(f"[STATE] {corr_obs}")
        state.raw_user_inputs["service_recency"] = recency
        state.entity_observations["service_recency"] = recency
        state_changed = True

    # 4c. Store temporal candidates
    temp_cands = extraction.get("temporal_candidates", {})
    if isinstance(temp_cands, dict):
        for k, v in temp_cands.items():
            state.temporal_candidates[k] = v
            state.raw_user_inputs.setdefault("temporal_candidates", {})[k] = v

    # 5. Store ambiguities and handle active ambiguity tracking
    raw_ambiguities = extraction.get("ambiguities", [])
    if isinstance(raw_ambiguities, list):
        for amb in raw_ambiguities:
            ambiguities.append(amb)
            state.raw_user_inputs.setdefault("ambiguities", []).append(amb)

    active_clarification = next(
        (a for a in raw_ambiguities if isinstance(a, dict) and a.get("type") in ("active_question_timing_clarification", "subjective_condition_ambiguity")),
        None
    )
    if active_clarification:
        prev_amb = state.active_ambiguity
        if prev_amb and prev_amb.get("feature") == active_clarification.get("feature"):
            attempt = prev_amb.get("attempt", 1) + 1
        else:
            attempt = 1
        active_clarification["attempt"] = attempt
        state.active_ambiguity = active_clarification
        state_changed = True
    else:
        # If no clarification occurred in this turn, clear any lingering active_ambiguity
        if state.active_ambiguity is not None:
            state.active_ambiguity = None
            state_changed = True

    # 6. Store approximate features
    approx_dict = extraction.get("approximate_features", {})
    if isinstance(approx_dict, dict):
        state.raw_user_inputs.setdefault("approximate_features", {}).update(approx_dict)

    # 7. Store raw observations
    raw_obs = extraction.get("raw_observations", [])
    if isinstance(raw_obs, list):
        for obs in raw_obs:
            if obs and isinstance(obs, str) and obs not in state.raw_observations:
                state.raw_observations.append(obs)

    return {
        "state_changed": state_changed,
        "updated_features": updated_features,
        "declined_features": declined_features,
        "rejected_features": rejected_features,
        "contradictions": contradictions,
        "ambiguities": ambiguities,
        "symptoms": state.symptoms,
        "replacement_history": state.raw_user_inputs.get("replacement_history", {}),
        "pending_features": state.raw_user_inputs.get("pending_features", []),
        "temporal_candidates": state.temporal_candidates,
        "entity_observations": state.entity_observations,
        "known_count": state.known_feature_count(),
    }
