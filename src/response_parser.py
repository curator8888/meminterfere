"""
MemInterfere: Robust Response Parser

Parses LLM responses through a cascade of strategies:
1. Structured JSON extraction
2. Regex pattern matching (TOOL: / CONFIDENCE: / APPROACH:)
3. Freeform skill name matching against the known skill library
4. Tags unparseable responses as PARSE_ERROR (never silently drops to 0%)

This replaces the brittle single-format parser that caused Llama-3.1-8b's
0% parse rate in Phase 4.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Parse method tags ────────────────────────────────────────────────────────

PARSE_METHOD_JSON = "json"
PARSE_METHOD_REGEX = "regex"
PARSE_METHOD_FREEFORM = "freeform"
PARSE_METHOD_ERROR = "parse_error"


@dataclass
class ParsedResponse:
    """Result of parsing an LLM response."""
    tool_name: str          # The identified skill/tool name, or "PARSE_ERROR"
    confidence: float       # 0.0-1.0, or 0.0 if unknown
    approach: str           # Brief description of approach, or ""
    parse_method: str       # One of PARSE_METHOD_* constants
    raw_response: str       # Original response text
    all_matched_skills: list[str] = field(default_factory=list)  # All skills found in response


@dataclass
class ParseStats:
    """Tracks statistics across many parsed responses."""
    json_count: int = 0
    regex_count: int = 0
    freeform_count: int = 0
    error_count: int = 0
    total_count: int = 0

    def record(self, method: str):
        self.total_count += 1
        if method == PARSE_METHOD_JSON:
            self.json_count += 1
        elif method == PARSE_METHOD_REGEX:
            self.regex_count += 1
        elif method == PARSE_METHOD_FREEFORM:
            self.freeform_count += 1
        elif method == PARSE_METHOD_ERROR:
            self.error_count += 1

    @property
    def parse_rate(self) -> float:
        """Fraction of responses that were successfully parsed (not error)."""
        if self.total_count == 0:
            return 0.0
        return (self.total_count - self.error_count) / self.total_count

    @property
    def json_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.json_count / self.total_count

    @property
    def regex_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.regex_count / self.total_count

    @property
    def freeform_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.freeform_count / self.total_count

    def summary(self) -> dict:
        return {
            "total": self.total_count,
            "json": self.json_count,
            "regex": self.regex_count,
            "freeform": self.freeform_count,
            "error": self.error_count,
            "parse_rate": round(self.parse_rate, 4),
            "json_rate": round(self.json_rate, 4),
            "regex_rate": round(self.regex_rate, 4),
            "freeform_rate": round(self.freeform_rate, 4),
        }


# ── Parsing functions ────────────────────────────────────────────────────────

def _try_json_extraction(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from the response.
    Handles:
    - Pure JSON response
    - JSON embedded in markdown code blocks
    - JSON with leading/trailing text
    """
    # 1. Try parsing the whole text as JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 2. Try extracting JSON from markdown code blocks
    md_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    matches = re.findall(md_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    # 3. Try finding the first { ... } block
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    return None


def _normalize_skill_name(name: str) -> str:
    """Normalize a skill name for matching: lowercase, strip whitespace and punctuation."""
    return re.sub(r'[^a-z0-9_]', '', name.lower().strip())


def parse_response(
    response: str,
    valid_skill_names: Optional[list[str]] = None,
) -> ParsedResponse:
    """
    Parse an LLM response using a cascade of strategies.

    Args:
        response: The raw LLM response text.
        valid_skill_names: Optional list of known valid skill names for freeform matching.
                          If provided, enables freeform extraction fallback.

    Returns:
        ParsedResponse with the parsed data and method used.
    """
    if not response or not response.strip():
        return ParsedResponse(
            tool_name="PARSE_ERROR",
            confidence=0.0,
            approach="",
            parse_method=PARSE_METHOD_ERROR,
            raw_response=response,
        )

    # Build normalized skill name lookup if provided
    normalized_skills = {}
    if valid_skill_names:
        for name in valid_skill_names:
            normalized_skills[_normalize_skill_name(name)] = name

    # ── Strategy 1: JSON extraction ──────────────────────────────────────
    json_data = _try_json_extraction(response)
    if json_data:
        tool_name = ""
        confidence = 0.0
        approach = ""

        # Try common field names for tool/skill name
        for key in ["tool", "tool_name", "skill", "skill_name", "action", "function",
                     "TOOL", "TOOL_NAME", "SKILL", "SKILL_NAME"]:
            if key in json_data:
                tool_name = str(json_data[key]).strip()
                break

        # Try common field names for confidence
        for key in ["confidence", "CONFIDENCE", "conf", "score", "certainty"]:
            if key in json_data:
                try:
                    conf = float(json_data[key])
                    confidence = max(0.0, min(1.0, conf))
                except (ValueError, TypeError):
                    pass
                break

        # Try common field names for approach
        for key in ["approach", "APPROACH", "reasoning", "description", "plan", "explanation"]:
            if key in json_data:
                approach = str(json_data[key]).strip()
                break

        # Validate tool_name against known skills if available
        if valid_skill_names and tool_name:
            norm = _normalize_skill_name(tool_name)
            if norm in normalized_skills:
                tool_name = normalized_skills[norm]

        if tool_name:
            return ParsedResponse(
                tool_name=tool_name,
                confidence=confidence,
                approach=approach,
                parse_method=PARSE_METHOD_JSON,
                raw_response=response,
                all_matched_skills=_find_all_skills(response, normalized_skills),
            )

    # ── Strategy 2: Regex pattern matching ────────────────────────────────
    # Match patterns like: TOOL: search_web, CONFIDENCE: 0.85, APPROACH: ...
    tool_match = re.search(r'(?:TOOL|SKILL|FUNCTION|ACTION)\s*[:=]\s*["\']?(\w[\w\s]*?)["\']?(?:\s*[,;\n]|$)',
                           response, re.IGNORECASE)
    conf_match = re.search(r'(?:CONFIDENCE|CONF|SCORE|CERTAINTY)\s*[:=]\s*["\']?([0-9]*\.?[0-9]+)["\']?',
                           response, re.IGNORECASE)
    approach_match = re.search(r'(?:APPROACH|REASONING|DESCRIPTION|PLAN|EXPLANATION)\s*[:=]\s*["\']?(.+?)(?:["\']?\s*(?:\n|$))',
                               response, re.IGNORECASE | re.DOTALL)

    if tool_match:
        tool_name = tool_match.group(1).strip()
        confidence = 0.0
        approach = ""

        if conf_match:
            try:
                confidence = max(0.0, min(1.0, float(conf_match.group(1))))
            except ValueError:
                pass

        if approach_match:
            approach = approach_match.group(1).strip()

        # Validate tool_name against known skills if available
        if valid_skill_names:
            norm = _normalize_skill_name(tool_name)
            if norm in normalized_skills:
                tool_name = normalized_skills[norm]

        return ParsedResponse(
            tool_name=tool_name,
            confidence=confidence,
            approach=approach,
            parse_method=PARSE_METHOD_REGEX,
            raw_response=response,
            all_matched_skills=_find_all_skills(response, normalized_skills),
        )

    # ── Strategy 3: Freeform skill name matching ──────────────────────────
    if valid_skill_names:
        found_skills = _find_all_skills(response, normalized_skills)
        if found_skills:
            # Use the first found skill as the primary tool
            tool_name = found_skills[0]
            # Try to extract confidence if present anywhere
            conf_match = re.search(r'([0-9]*\.?[0-9]+)\s*(?:confidence|confident|certain|sure)',
                                   response, re.IGNORECASE)
            confidence = 0.5  # Default moderate confidence for freeform
            if conf_match:
                try:
                    confidence = max(0.0, min(1.0, float(conf_match.group(1))))
                except ValueError:
                    pass

            return ParsedResponse(
                tool_name=tool_name,
                confidence=confidence,
                approach="",
                parse_method=PARSE_METHOD_FREEFORM,
                raw_response=response,
                all_matched_skills=found_skills,
            )

    # ── Fallback: PARSE_ERROR ─────────────────────────────────────────────
    return ParsedResponse(
        tool_name="PARSE_ERROR",
        confidence=0.0,
        approach="",
        parse_method=PARSE_METHOD_ERROR,
        raw_response=response,
        all_matched_skills=_find_all_skills(response, normalized_skills) if valid_skill_names else [],
    )


def _find_all_skills(text: str, normalized_skills: dict[str, str]) -> list[str]:
    """Find all occurrences of known skill names in text (order of appearance)."""
    found = []
    text_lower = text.lower()
    for norm_name, orig_name in normalized_skills.items():
        # Check for the skill name as a whole word/phrase
        if norm_name in _normalize_skill_name(text_lower) or orig_name.lower() in text_lower:
            if orig_name not in found:
                found.append(orig_name)
    return found


def batch_parse(responses: list[str], valid_skill_names: Optional[list[str]] = None) -> tuple[list[ParsedResponse], ParseStats]:
    """
    Parse a batch of responses and return results with statistics.

    Args:
        responses: List of raw LLM responses.
        valid_skill_names: Optional list of known valid skill names.

    Returns:
        Tuple of (list of ParsedResponse, ParseStats)
    """
    results = []
    stats = ParseStats()
    for response in responses:
        parsed = parse_response(response, valid_skill_names)
        results.append(parsed)
        stats.record(parsed.parse_method)
    return results, stats


# ── Test / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example skill names from the library
    sample_skills = [
        "search_web", "navigate_url", "scrape_page", "extract_links",
        "take_screenshot", "fill_form", "download_file", "translate_page",
        "summarize_page", "monitor_page", "crawl_site", "extract_table",
        "check_website_status", "read_rss", "extract_metadata",
        "create_event", "list_events", "send_email", "read_emails",
        "read_file", "get_weather", "get_stock_price", "calculate",
        "create_task", "list_tasks", "query_database", "write_file",
        "lookup_contact", "set_reminder", "search_news",
        # Interference skills
        "find_information", "browse_website", "get_data_from_page",
        "quick_summary", "add_event", "compose_email", "fetch_emails",
        "read_document", "check_weather", "get_stock_quote", "compute",
        "create_todo", "view_tasks", "sql_query", "save_file",
        "find_contact", "schedule_reminder", "fetch_news",
    ]

    test_responses = [
        # JSON response
        '{"tool": "search_web", "confidence": 0.9, "approach": "Use search to find results"}',
        # Markdown-wrapped JSON
        '```json\n{"TOOL_NAME": "navigate_url", "CONFIDENCE": 0.85, "APPROACH": "Navigate directly"}\n```',
        # Regex-style response
        'TOOL: scrape_page\nCONFIDENCE: 0.75\nAPPROACH: Extract data using CSS selectors',
        # Freeform response mentioning a skill
        'I would use the search_web skill to find information about machine learning tutorials.',
        # Completely unparseable
        'I think we should probably look at the website and see what happens.',
    ]

    print("Response Parser Demo")
    print("=" * 80)
    results, stats = batch_parse(test_responses, sample_skills)
    for i, (resp, parsed) in enumerate(zip(test_responses, results)):
        print(f"\n--- Response {i+1} ---")
        print(f"  Raw: {resp[:80]}...")
        print(f"  Method: {parsed.parse_method}")
        print(f"  Tool: {parsed.tool_name}")
        print(f"  Confidence: {parsed.confidence}")
        print(f"  Approach: {parsed.approach[:60]}...")
        print(f"  All matched: {parsed.all_matched_skills}")

    print(f"\n{'=' * 80}")
    print(f"Parse Statistics: {stats.summary()}")