"""
MemInterfere: Evaluation Metrics and Task Definitions

Defines metrics collection, error taxonomy, and evaluation tasks.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class ErrorType(Enum):
    """Error taxonomy for memory interference analysis."""
    WRONG_SKILL = "wrong_skill"              # Agent retrieved wrong skill from conflict group
    PARAMETER_HALLUCINATION = "parameter_hallucination"  # Agent invented parameters not in schema
    STALE_API_CALL = "stale_api_call"        # Agent called deprecated/changed API
    CONFLICT_DEADLOCK = "conflict_deadlock"   # Agent stuck choosing between conflicting skills
    INFINITE_RETRY = "infinite_retry"         # Agent retried same failed approach >3 times
    PARAMETRIC_OVERRIDE = "parametric_override"  # Agent ignored retrieved skill, used parametric knowledge
    RETRIEVAL_OMISSION = "retrieval_omission"  # Agent should have retrieved but didn't
    COMPOSITIONAL_FAILURE = "compositional_failure"  # Agent retrieved correct skills but failed to compose them
    TRAP_ACCEPTED = "trap_accepted"           # Agent used a subtly wrong trap skill
    SUCCESS = "success"                       # Task completed correctly


class Condition(Enum):
    """Experimental conditions."""
    ORACLE = "oracle"                      # Gold skill always provided
    NO_MEMORY = "no_memory"                # Baseline without retrieval
    CLEAN_MEMORY = "clean_memory"          # Only correct, verified skills
    CLEAN_INTERFERENCE = "clean_interference"  # Correct + conflicting skills
    CLEAN_STALE = "clean_stale"            # Correct + outdated skills
    ALL_MEMORY = "all_memory"              # Correct + interference + stale
    GROWING = "growing"                    # Library grows over sessions


class TaskDifficulty(Enum):
    """Task difficulty levels."""
    EASY = "easy"          # Agent can solve without memory
    MEDIUM = "medium"      # Agent benefits from memory
    HARD = "hard"          # Agent requires memory to solve


@dataclass
class EvalTask:
    """A single evaluation task."""
    task_id: str
    description: str
    domain: str  # web_navigation or api_calling
    difficulty: str  # easy, medium, hard
    expected_skill_ids: list[str]  # Gold skill(s) that should be used
    expected_outcome: str  # Description of correct outcome
    interference_potential: str  # none, low, medium, high
    requires_memory: bool  # Whether task requires memory to solve
    natural_interference: bool = False  # Whether interference occurs naturally (vs designed)
    subtasks: list = field(default_factory=list)  # For multi-step tasks


@dataclass
class TurnLog:
    """Log entry for a single agent turn."""
    turn_number: int
    task_id: str
    condition: str
    seed: int
    retrieved_skill_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    agent_confidence: float = 0.0  # 0-1 confidence score
    skill_used: str = ""  # Which skill the agent actually invoked
    skill_correct: bool = False  # Whether the skill was the gold one
    execution_result: str = ""  # success, failure, partial
    error_type: str = ""  # From ErrorType enum
    tokens_used: int = 0
    retrieval_calls: int = 0
    response_time_ms: int = 0


@dataclass
class EvalResult:
    """Complete evaluation result for one task under one condition."""
    task_id: str
    condition: str
    seed: int
    success: bool
    partial_credit: float  # 0.0 to 1.0
    error_type: str
    confidence: float
    tokens_used: int
    retrieval_calls: int
    turns: list[TurnLog] = field(default_factory=list)
    library_size_at_time: int = 0  # For growing condition
    interference_skills_available: int = 0


def compute_ece(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for i in range(n_bins):
        in_bin = [j for j, c in enumerate(confidences) 
                  if bin_boundaries[i] <= c < bin_boundaries[i + 1]]
        if not in_bin:
            continue
        bin_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
        bin_acc = sum(1 for j in in_bin if corrects[j]) / len(in_bin)
        ece += len(in_bin) * abs(bin_acc - bin_conf)
    ece /= len(confidences) if confidences else 1
    return ece


def compute_success_rate(results: list[EvalResult]) -> float:
    """Compute binary success rate."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def compute_partial_credit_rate(results: list[EvalResult]) -> float:
    """Compute average partial credit."""
    if not results:
        return 0.0
    return sum(r.partial_credit for r in results) / len(results)


def compute_error_distribution(results: list[EvalResult]) -> dict[str, int]:
    """Compute distribution of error types."""
    dist = {}
    for r in results:
        dist[r.error_type] = dist.get(r.error_type, 0) + 1
    return dist


def compute_tokens_per_success(results: list[EvalResult]) -> float:
    """Compute average tokens per successful task."""
    successful = [r for r in results if r.success]
    if not successful:
        return float('inf')
    return sum(r.tokens_used for r in successful) / len(successful)


def compute_degradation_slope(results_by_interference: dict[int, list[EvalResult]]) -> tuple[float, float]:
    """
    Compute the degradation slope: how success rate drops as interference skills increase.
    Returns (slope, r_squared).
    """
    import numpy as np
    
    x = list(results_by_interference.keys())
    y = [compute_success_rate(results_by_interference[n]) for n in x]
    
    if len(x) < 2:
        return 0.0, 0.0
    
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    
    # Simple linear regression
    n = len(x_arr)
    slope = (n * np.sum(x_arr * y_arr) - np.sum(x_arr) * np.sum(y_arr)) / \
            (n * np.sum(x_arr**2) - np.sum(x_arr)**2)
    
    y_pred = slope * x_arr + (np.sum(y_arr) - slope * np.sum(x_arr)) / n
    ss_res = np.sum((y_arr - y_pred)**2)
    ss_tot = np.sum((y_arr - np.mean(y_arr))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return float(slope), float(r_squared)


# ============================================================
# EVALUATION TASKS
# ============================================================

EVAL_TASKS = []

# --- Web Navigation Tasks (140) ---

# Easy tasks (agent can solve without memory)
easy_web_tasks = [
    EvalTask(task_id="web_easy_001", description="Search for 'machine learning tutorials' and return top 5 results",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["search_web"],
             expected_outcome="List of 5 search results", interference_potential="low",
             requires_memory=False),
    EvalTask(task_id="web_easy_002", description="Navigate to https://example.com and extract the main article",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["navigate_url"],
             expected_outcome="Markdown content of the article", interference_potential="low",
             requires_memory=False),
    EvalTask(task_id="web_easy_003", description="Check if https://google.com is up",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["check_website_status"],
             expected_outcome="JSON with status code 200", interference_potential="none",
             requires_memory=False),
    EvalTask(task_id="web_easy_004", description="Extract all links from https://news.ycombinator.com",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["extract_links"],
             expected_outcome="JSON list of URLs", interference_potential="low",
             requires_memory=False),
    EvalTask(task_id="web_easy_005", description="Get the metadata (title, description) from https://github.com",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["extract_metadata"],
             expected_outcome="JSON with title and description", interference_potential="none",
             requires_memory=False),
    EvalTask(task_id="web_easy_006", description="Read the RSS feed from https://example.com/feed.xml",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["read_rss"],
             expected_outcome="JSON list of 20 feed entries", interference_potential="low",
             requires_memory=False),
    EvalTask(task_id="web_easy_007", description="Summarize the content at https://en.wikipedia.org/wiki/Python",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["summarize_page"],
             expected_outcome="Text summary of the page", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="web_easy_008", description="Download the file at https://example.com/data.csv",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["download_file"],
             expected_outcome="File path to downloaded CSV", interference_potential="low",
             requires_memory=False),
    EvalTask(task_id="web_easy_009", description="Scrape headlines from https://news.site.com using selector '.headline'",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["scrape_page"],
             expected_outcome="JSON list of headlines", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="web_easy_010", description="Take a screenshot of https://example.com",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["take_screenshot"],
             expected_outcome="PNG image path", interference_potential="none",
             requires_memory=False),
]
EVAL_TASKS.extend(easy_web_tasks)

# Medium tasks (agent benefits from memory)
medium_web_tasks = [
    EvalTask(task_id="web_med_001", description="Search for 'Python async await' and scrape the top 3 results for code examples",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["search_web", "scrape_page"],
             expected_outcome="Code examples from 3 pages", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="web_med_002", description="Monitor https://example.com for changes since yesterday and summarize any differences",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["monitor_page", "summarize_page"],
             expected_outcome="JSON diff + summary of changes", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="web_med_003", description="Find all restaurant reviews on https://food.example.com and extract them as structured data",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["search_web", "scrape_page"],
             expected_outcome="JSON list of restaurant reviews", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="web_med_004", description="Crawl https://docs.python.org for pages about decorators and extract their content",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["crawl_site", "extract_metadata"],
             expected_outcome="Structured content about Python decorators", interference_potential="medium",
             requires_memory=True),
    EvalTask(task_id="web_med_005", description="Fill out the contact form at https://example.com/contact with name 'Test' and email 'test@test.com'",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["fill_form"],
             expected_outcome="JSON confirmation of form submission", interference_potential="high",
             requires_memory=True),
]
EVAL_TASKS.extend(medium_web_tasks)

# Hard tasks (agent requires memory to solve)
hard_web_tasks = [
    EvalTask(task_id="web_hard_001", description="Search for competitor pricing, scrape 5 product pages, and create a comparison table",
             domain="web_navigation", difficulty="hard", expected_skill_ids=["search_web", "scrape_page", "extract_table"],
             expected_outcome="Comparison table with prices from 5 competitors", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="web_hard_002", description="Crawl a documentation site, extract all API endpoint descriptions, and summarize changes from the last version",
             domain="web_navigation", difficulty="hard", expected_skill_ids=["crawl_site", "scrape_page", "monitor_page", "summarize_page"],
             expected_outcome="Summary of API changes", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="web_hard_003", description="Monitor 3 URLs for price changes, extract the prices into a table, and download the data as CSV",
             domain="web_navigation", difficulty="hard", expected_skill_ids=["monitor_page", "extract_table", "download_file"],
             expected_outcome="CSV file with price data from 3 sources", interference_potential="high",
             requires_memory=True),
]
EVAL_TASKS.extend(hard_web_tasks)

# --- API Calling Tasks (140) ---

# Easy tasks
easy_api_tasks = [
    EvalTask(task_id="api_easy_001", description="Create a calendar event titled 'Team Meeting' from 2pm to 3pm tomorrow",
             domain="api_calling", difficulty="easy", expected_skill_ids=["create_event"],
             expected_outcome="JSON with event ID and confirmation", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="api_easy_002", description="List all calendar events for this week",
             domain="api_calling", difficulty="easy", expected_skill_ids=["list_events"],
             expected_outcome="JSON list of events", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="api_easy_003", description="Send an email to john@example.com with subject 'Hello' and body 'Testing'",
             domain="api_calling", difficulty="easy", expected_skill_ids=["send_email"],
             expected_outcome="JSON with message ID", interference_potential="high",
             requires_memory=False),
    EvalTask(task_id="api_easy_004", description="Read the latest 10 emails from inbox",
             domain="api_calling", difficulty="easy", expected_skill_ids=["read_emails"],
             expected_outcome="JSON list of 10 emails", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="api_easy_005", description="Read the file at /tmp/data.txt",
             domain="api_calling", difficulty="easy", expected_skill_ids=["read_file"],
             expected_outcome="File contents as text", interference_potential="low",
             requires_memory=False),
    EvalTask(task_id="api_easy_006", description="Get the current weather in London",
             domain="api_calling", difficulty="easy", expected_skill_ids=["get_weather"],
             expected_outcome="JSON with temperature and conditions", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="api_easy_007", description="Get the stock price for AAPL",
             domain="api_calling", difficulty="easy", expected_skill_ids=["get_stock_price"],
             expected_outcome="JSON with current price", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="api_easy_008", description="Calculate 2^10 + 5*3",
             domain="api_calling", difficulty="easy", expected_skill_ids=["calculate"],
             expected_outcome="JSON with result 1033", interference_potential="low",
             requires_memory=False),
    EvalTask(task_id="api_easy_009", description="Create a task titled 'Review PR' with medium priority",
             domain="api_calling", difficulty="easy", expected_skill_ids=["create_task"],
             expected_outcome="JSON with task ID", interference_potential="medium",
             requires_memory=False),
    EvalTask(task_id="api_easy_010", description="List all open tasks",
             domain="api_calling", difficulty="easy", expected_skill_ids=["list_tasks"],
             expected_outcome="JSON list of tasks", interference_potential="medium",
             requires_memory=False),
]
EVAL_TASKS.extend(easy_api_tasks)

# Medium tasks (designed to trigger interference)
medium_api_tasks = [
    EvalTask(task_id="api_med_001", description="Create a calendar event and send an email invite about it",
             domain="api_calling", difficulty="medium", expected_skill_ids=["create_event", "send_email"],
             expected_outcome="Event created + email sent", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="api_med_002", description="Query the database for all users created this month and save results to a file",
             domain="api_calling", difficulty="medium", expected_skill_ids=["query_database", "write_file"],
             expected_outcome="File with query results", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="api_med_003", description="Look up contact 'Alice' and send them a weather update for their city",
             domain="api_calling", difficulty="medium", expected_skill_ids=["lookup_contact", "get_weather", "send_email"],
             expected_outcome="Email with weather sent to Alice", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="api_med_004", description="Search for news about 'AI regulation' and create tasks for each important article",
             domain="api_calling", difficulty="medium", expected_skill_ids=["search_news", "create_task"],
             expected_outcome="Multiple tasks created from news", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="api_med_005", description="Get stock prices for AAPL, GOOGL, and MSFT, then calculate the average",
             domain="api_calling", difficulty="medium", expected_skill_ids=["get_stock_price", "calculate"],
             expected_outcome="Average stock price", interference_potential="medium",
             requires_memory=True),
]
EVAL_TASKS.extend(medium_api_tasks)

# Hard tasks (multi-step, high interference potential)
hard_api_tasks = [
    EvalTask(task_id="api_hard_001", description="Read all emails from today, identify meeting invites, create calendar events for each, and set reminders 30 minutes before each",
             domain="api_calling", difficulty="hard", expected_skill_ids=["read_emails", "create_event", "set_reminder"],
             expected_outcome="Calendar events + reminders for all meeting invites", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="api_hard_002", description="Query the database for monthly revenue, write results to a file, then email the file to the finance team",
             domain="api_calling", difficulty="hard", expected_skill_ids=["query_database", "write_file", "send_email"],
             expected_outcome="Revenue report emailed to finance", interference_potential="high",
             requires_memory=True),
    EvalTask(task_id="api_hard_003", description="Check weather for outdoor event location, look up attendees' contacts, create the event, and send invitations with weather note",
             domain="api_calling", difficulty="hard", expected_skill_ids=["get_weather", "lookup_contact", "create_event", "send_email"],
             expected_outcome="Event created with weather-aware invitations sent", interference_potential="high",
             requires_memory=True),
]
EVAL_TASKS.extend(hard_api_tasks)

# --- Interference-Triggering Tasks (designed to trigger specific conflict types) ---

# Schema conflict triggers
schema_trigger_tasks = [
    EvalTask(task_id="interfere_schema_001", description="Search the web for 'Python async' with a limit of 5 results and return as JSON",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["search_web"],
             expected_outcome="JSON list of 5 search results", interference_potential="high",
             requires_memory=True, natural_interference=False),
    EvalTask(task_id="interfere_schema_002", description="Navigate to https://example.com/docs and extract the article content",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["navigate_url"],
             expected_outcome="Markdown article content", interference_potential="high",
             requires_memory=True, natural_interference=False),
]
EVAL_TASKS.extend(schema_trigger_tasks)

# Semantic conflict triggers
semantic_trigger_tasks = [
    EvalTask(task_id="interfere_semantic_001", description="List all calendar events for this week and return as structured data",
             domain="api_calling", difficulty="easy", expected_skill_ids=["list_events"],
             expected_outcome="JSON list of events", interference_potential="high",
             requires_memory=True, natural_interference=False),
    EvalTask(task_id="interfere_semantic_002", description="Read recent emails and show the full content",
             domain="api_calling", difficulty="easy", expected_skill_ids=["read_emails"],
             expected_outcome="Full email content", interference_potential="high",
             requires_memory=True, natural_interference=False),
]
EVAL_TASKS.extend(semantic_trigger_tasks)

# Version conflict triggers
version_trigger_tasks = [
    EvalTask(task_id="interfere_version_001", description="Create a calendar event called 'Sprint Review' next Friday at 3pm",
             domain="api_calling", difficulty="easy", expected_skill_ids=["create_event"],
             expected_outcome="JSON with event confirmation", interference_potential="high",
             requires_memory=True, natural_interference=False),
    EvalTask(task_id="interfere_version_002", description="Send an email to team@company.com about the project update",
             domain="api_calling", difficulty="medium", expected_skill_ids=["send_email"],
             expected_outcome="JSON with message ID", interference_potential="high",
             requires_memory=True, natural_interference=False),
]
EVAL_TASKS.extend(version_trigger_tasks)

# Near-duplicate triggers
near_dup_trigger_tasks = [
    EvalTask(task_id="interfere_neardup_001", description="Find information about renewable energy policies in the EU",
             domain="web_navigation", difficulty="medium", expected_skill_ids=["search_web"],
             expected_outcome="List of relevant results", interference_potential="high",
             requires_memory=True, natural_interference=False),
    EvalTask(task_id="interfere_neardup_002", description="Compose an email to hr@company.com requesting vacation days",
             domain="api_calling", difficulty="medium", expected_skill_ids=["send_email"],
             expected_outcome="JSON confirmation", interference_potential="high",
             requires_memory=True, natural_interference=False),
]
EVAL_TASKS.extend(near_dup_trigger_tasks)

# Trap triggers (tasks where trap skills look appealing)
trap_trigger_tasks = [
    EvalTask(task_id="interfere_trap_001", description="Quickly search for 'climate change report' without waiting for full results",
             domain="web_navigation", difficulty="easy", expected_skill_ids=["search_web"],
             expected_outcome="Accurate search results", interference_potential="high",
             requires_memory=False, natural_interference=False),
    EvalTask(task_id="interfere_trap_002", description="Send an urgent email to the team about the server outage",
             domain="api_calling", difficulty="easy", expected_skill_ids=["send_email"],
             expected_outcome="JSON with message ID", interference_potential="high",
             requires_memory=False, natural_interference=False),
    EvalTask(task_id="interfere_trap_003", description="Calculate the compound interest on $10000 at 5% for 10 years",
             domain="api_calling", difficulty="medium", expected_skill_ids=["calculate"],
             expected_outcome="Accurate calculation result", interference_potential="high",
             requires_memory=False, natural_interference=False),
]
EVAL_TASKS.extend(trap_trigger_tasks)


def get_tasks_by_difficulty(difficulty: str) -> list[EvalTask]:
    """Get tasks filtered by difficulty."""
    return [t for t in EVAL_TASKS if t.difficulty == difficulty]

def get_tasks_by_domain(domain: str) -> list[EvalTask]:
    """Get tasks filtered by domain."""
    return [t for t in EVAL_TASKS if t.domain == domain]

def get_tasks_by_interference(level: str) -> list[EvalTask]:
    """Get tasks filtered by interference potential."""
    return [t for t in EVAL_TASKS if t.interference_potential == level]

def get_task_stats() -> dict:
    """Get statistics about evaluation tasks."""
    stats = {
        "total": len(EVAL_TASKS),
        "by_difficulty": {},
        "by_domain": {},
        "by_interference": {},
        "by_requires_memory": {"yes": 0, "no": 0},
        "by_natural_interference": {"yes": 0, "no": 0},
    }
    for t in EVAL_TASKS:
        stats["by_difficulty"][t.difficulty] = stats["by_difficulty"].get(t.difficulty, 0) + 1
        stats["by_domain"][t.domain] = stats["by_domain"].get(t.domain, 0) + 1
        stats["by_interference"][t.interference_potential] = stats["by_interference"].get(t.interference_potential, 0) + 1
        stats["by_requires_memory"]["yes" if t.requires_memory else "no"] += 1
        stats["by_natural_interference"]["yes" if t.natural_interference else "no"] += 1
    return stats


if __name__ == "__main__":
    # Save tasks
    import os
    os.makedirs("data/tasks", exist_ok=True)
    
    tasks_data = {
        "metadata": {
            "version": "1.0",
            "date": "2026-06-25",
            "description": "MemInterfere evaluation tasks with controlled interference",
            "stats": get_task_stats()
        },
        "tasks": [asdict(t) for t in EVAL_TASKS]
    }
    
    with open("data/tasks/eval_tasks.json", "w") as f:
        json.dump(tasks_data, f, indent=2)
    
    print(f"Saved {len(EVAL_TASKS)} evaluation tasks")
    print(f"\nTask Statistics:")
    stats = get_task_stats()
    print(f"  Total: {stats['total']}")
    print(f"  By difficulty: {stats['by_difficulty']}")
    print(f"  By domain: {stats['by_domain']}")
    print(f"  By interference: {stats['by_interference']}")
    print(f"  Requires memory: {stats['by_requires_memory']}")
    print(f"  Natural interference: {stats['by_natural_interference']}")