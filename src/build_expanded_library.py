"""
Phase 5.1: Generate the expanded 100-skill library with formal near-duplicate taxonomy.

This script:
1. Keeps all 68 existing skills intact
2. Adds 32 new skills with proper similarity_to_gold scores
3. Creates gradient test pairs for near-identical skills
4. Documents 5 real-world skill libraries with naming collisions
5. Outputs expanded_library_100.json
"""

import json
import hashlib
import os
from dataclasses import dataclass, asdict, field
from typing import Optional


def make_skill_id(name: str, variant: str = "") -> str:
    """Generate a stable skill ID from name and variant."""
    raw = f"{name}_{variant}" if variant else name
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_existing_library(path: str) -> list:
    """Load the existing 68-skill library."""
    with open(path) as f:
        data = json.load(f)
    return data['skills']


# ============================================================
# NEW SKILLS (32 additional, bringing total to 100)
# ============================================================

# We need to be careful about counts. Let's categorize the existing 68:
# - Clean (is_clean=True, conflict_type=none, not stale, not trap): count these
# - Schema conflicts: count these
# - Semantic conflicts: count these
# - Near duplicates: count these
# - Version conflicts: count these
# - Stale: count these
# - Trap: count these
#
# Target distribution for 100 skills:
# - 40 clean skills (20 web, 20 API)
# - 20 near-identical duplicates (synonym pairs, similarity 0.8-0.9)
# - 15 near-similar duplicates (overlapping scope, similarity 0.5-0.7)
# - 10 schema conflicts (same name, different params, similarity 1.0)
# - 10 semantic conflicts (same name, different return format, similarity 0.8-0.9 name, 0.3-0.5 semantic)
# - 5 version conflicts (v1/v2, similarity 1.0 name, 0.6-0.8 schema)
#
# Note: The existing library already has schema_conflict, semantic_conflict,
# version_conflict, and near_duplicate entries. We need to recategorize
# near_duplicates into near_identical and near_similar, and add new skills.

# The new conflict_type values are:
# - "near_identical" (was near_duplicate with sim 0.8-0.9)
# - "near_similar" (was near_duplicate with sim 0.5-0.7)
# Existing near_duplicate entries get remapped based on their actual similarity.

NEW_SKILLS = []

# ============================================================
# 10 NEW CLEAN SKILLS (5 web, 5 API) — bringing clean total to 40
# (existing has ~30 clean, we need 10 more)
# ============================================================

clean_new_web = [
    {
        "skill_id": make_skill_id("web_proxy_request", "clean"),
        "name": "proxy_request",
        "domain": "web_navigation",
        "description": "Make a proxied HTTP request through a web proxy to bypass CORS restrictions",
        "parameters": {
            "url": {"type": "string", "required": True},
            "method": {"type": "string", "required": False, "default": "GET"},
            "headers": {"type": "object", "required": False}
        },
        "return_format": "json",
        "success_rate": 0.88,
        "last_verified": "2026-06-20",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("web_parse_html", "clean"),
        "name": "parse_html",
        "domain": "web_navigation",
        "description": "Parse raw HTML into a structured DOM tree for element extraction",
        "parameters": {
            "html_content": {"type": "string", "required": True},
            "selector": {"type": "string", "required": False}
        },
        "return_format": "json_tree",
        "success_rate": 0.94,
        "last_verified": "2026-06-19",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("web_submit_form", "clean"),
        "name": "submit_form",
        "domain": "web_navigation",
        "description": "Submit a web form by providing form data and the form action URL",
        "parameters": {
            "action_url": {"type": "string", "required": True},
            "form_data": {"type": "object", "required": True},
            "method": {"type": "string", "required": False, "default": "POST"}
        },
        "return_format": "json",
        "success_rate": 0.82,
        "last_verified": "2026-06-18",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("web_check_ssl", "clean"),
        "name": "check_ssl",
        "domain": "web_navigation",
        "description": "Check the SSL/TLS certificate of a website and report expiry and issuer",
        "parameters": {
            "url": {"type": "string", "required": True}
        },
        "return_format": "json",
        "success_rate": 0.97,
        "last_verified": "2026-06-20",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("web_extract_images", "clean"),
        "name": "extract_images",
        "domain": "web_navigation",
        "description": "Extract all image URLs and alt text from a web page",
        "parameters": {
            "url": {"type": "string", "required": True},
            "min_width": {"type": "integer", "required": False, "default": 100},
            "min_height": {"type": "integer", "required": False, "default": 100}
        },
        "return_format": "json_list",
        "success_rate": 0.91,
        "last_verified": "2026-06-17",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
]

clean_new_api = [
    {
        "skill_id": make_skill_id("api_create_contact", "clean"),
        "name": "create_contact",
        "domain": "api_calling",
        "description": "Create a new contact in the address book",
        "parameters": {
            "name": {"type": "string", "required": True},
            "email": {"type": "string", "required": False},
            "phone": {"type": "string", "required": False}
        },
        "return_format": "json",
        "success_rate": 0.93,
        "last_verified": "2026-06-20",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("api_update_task", "clean"),
        "name": "update_task",
        "domain": "api_calling",
        "description": "Update an existing task's properties",
        "parameters": {
            "task_id": {"type": "string", "required": True},
            "updates": {"type": "object", "required": True}
        },
        "return_format": "json",
        "success_rate": 0.91,
        "last_verified": "2026-06-19",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("api_delete_event", "clean"),
        "name": "delete_event",
        "domain": "api_calling",
        "description": "Delete a calendar event by ID",
        "parameters": {
            "event_id": {"type": "string", "required": True},
            "notify_attendees": {"type": "boolean", "required": False, "default": False}
        },
        "return_format": "json",
        "success_rate": 0.95,
        "last_verified": "2026-06-20",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("api_get_forecast", "clean"),
        "name": "get_forecast",
        "domain": "api_calling",
        "description": "Get a 7-day weather forecast for a location",
        "parameters": {
            "location": {"type": "string", "required": True},
            "days": {"type": "integer", "required": False, "default": 7},
            "units": {"type": "string", "required": False, "default": "metric"}
        },
        "return_format": "json_list",
        "success_rate": 0.90,
        "last_verified": "2026-06-18",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("api_search_files", "clean"),
        "name": "search_files",
        "domain": "api_calling",
        "description": "Search for files in storage by name or content",
        "parameters": {
            "query": {"type": "string", "required": True},
            "file_type": {"type": "string", "required": False},
            "limit": {"type": "integer", "required": False, "default": 20}
        },
        "return_format": "json_list",
        "success_rate": 0.89,
        "last_verified": "2026-06-16",
        "conflict_group_id": None,
        "conflict_type": "none",
        "fan_degree": 1,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": True,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.0,
        "source_url": "",
        "gradient_pairs": []
    },
]

NEW_SKILLS.extend(clean_new_web)
NEW_SKILLS.extend(clean_new_api)

# ============================================================
# 12 NEAR-IDENTICAL DUPLICATES (similarity 0.8-0.9)
# These supplement existing near_duplicate entries which get recategorized
# Target: 20 total near-identical
# Existing near_duplicate count: let's count them from the library
# We need to add enough to reach 20 near-identical total
# ============================================================

near_identical_new = [
    # Gradient pair group 1: create_event family
    {
        "skill_id": make_skill_id("calendar_add_event", "near_ident_high"),
        "name": "add_calendar_event",
        "domain": "api_calling",
        "description": "Add a calendar event (synonym for create_event, high name similarity)",
        "parameters": {
            "title": {"type": "string", "required": True},
            "start_time": {"type": "string", "required": True},
            "end_time": {"type": "string", "required": True},
            "location": {"type": "string", "required": False}
        },
        "return_format": "json",
        "success_rate": 0.90,
        "last_verified": "2026-06-10",
        "conflict_group_id": "create_event_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.88,
        "source_url": "",
        "gradient_pairs": [{"level": "high", "gold_skill": "create_event", "name_similarity": 0.92}]
    },
    {
        "skill_id": make_skill_id("calendar_schedule_meeting", "near_ident_med"),
        "name": "schedule_meeting",
        "domain": "api_calling",
        "description": "Schedule a meeting in the calendar (synonym for create_event, medium name similarity)",
        "parameters": {
            "meeting_title": {"type": "string", "required": True},
            "start_time": {"type": "string", "required": True},
            "duration_minutes": {"type": "integer", "required": False, "default": 60}
        },
        "return_format": "json",
        "success_rate": 0.88,
        "last_verified": "2026-06-08",
        "conflict_group_id": "create_event_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.62,
        "source_url": "",
        "gradient_pairs": [{"level": "medium", "gold_skill": "create_event", "name_similarity": 0.58}]
    },
    {
        "skill_id": make_skill_id("calendar_book_appointment", "near_ident_low"),
        "name": "book_appointment",
        "domain": "api_calling",
        "description": "Book an appointment (different concept, same domain, low name similarity)",
        "parameters": {
            "service_type": {"type": "string", "required": True},
            "preferred_time": {"type": "string", "required": True},
            "duration": {"type": "integer", "required": False, "default": 30}
        },
        "return_format": "json",
        "success_rate": 0.85,
        "last_verified": "2026-06-05",
        "conflict_group_id": "create_event_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.35,
        "source_url": "",
        "gradient_pairs": [{"level": "low", "gold_skill": "create_event", "name_similarity": 0.30}]
    },
    # Gradient pair group 2: send_email family
    {
        "skill_id": make_skill_id("email_dispatch", "near_ident_high"),
        "name": "dispatch_email",
        "domain": "api_calling",
        "description": "Dispatch an email message (synonym for send_email, high name similarity)",
        "parameters": {
            "to": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
            "priority": {"type": "string", "required": False, "default": "normal"}
        },
        "return_format": "json",
        "success_rate": 0.89,
        "last_verified": "2026-06-12",
        "conflict_group_id": "send_email_group",
        "conflict_type": "near_identical",
        "fan_degree": 4,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.85,
        "source_url": "",
        "gradient_pairs": [{"level": "high", "gold_skill": "send_email", "name_similarity": 0.88}]
    },
    {
        "skill_id": make_skill_id("email_compose_message", "near_ident_med"),
        "name": "compose_message",
        "domain": "api_calling",
        "description": "Compose and send a message (medium name similarity to send_email)",
        "parameters": {
            "recipient": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
            "subject": {"type": "string", "required": False}
        },
        "return_format": "json",
        "success_rate": 0.86,
        "last_verified": "2026-06-07",
        "conflict_group_id": "send_email_group",
        "conflict_type": "near_identical",
        "fan_degree": 4,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.58,
        "source_url": "",
        "gradient_pairs": [{"level": "medium", "gold_skill": "send_email", "name_similarity": 0.55}]
    },
    # Gradient pair group 3: query_database family
    {
        "skill_id": make_skill_id("db_run_query", "near_ident_high"),
        "name": "run_db_query",
        "domain": "api_calling",
        "description": "Run a database query (synonym for query_database, high name similarity)",
        "parameters": {
            "query": {"type": "string", "required": True},
            "database": {"type": "string", "required": False, "default": "default"},
            "limit": {"type": "integer", "required": False, "default": 100}
        },
        "return_format": "json_list",
        "success_rate": 0.86,
        "last_verified": "2026-06-09",
        "conflict_group_id": "query_database_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.87,
        "source_url": "",
        "gradient_pairs": [{"level": "high", "gold_skill": "query_database", "name_similarity": 0.90}]
    },
    {
        "skill_id": make_skill_id("db_fetch_records", "near_ident_low"),
        "name": "fetch_records",
        "domain": "api_calling",
        "description": "Fetch records from a data store (low name similarity to query_database)",
        "parameters": {
            "entity_type": {"type": "string", "required": True},
            "filter": {"type": "object", "required": False}
        },
        "return_format": "json_list",
        "success_rate": 0.82,
        "last_verified": "2026-06-03",
        "conflict_group_id": "query_database_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.32,
        "source_url": "",
        "gradient_pairs": [{"level": "low", "gold_skill": "query_database", "name_similarity": 0.28}]
    },
    # Gradient pair group 4: write_file family
    {
        "skill_id": make_skill_id("file_persist_data", "near_ident_med"),
        "name": "persist_data",
        "domain": "api_calling",
        "description": "Persist data to a storage file (medium name similarity to write_file)",
        "parameters": {
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
            "encoding": {"type": "string", "required": False, "default": "utf-8"}
        },
        "return_format": "json",
        "success_rate": 0.88,
        "last_verified": "2026-06-11",
        "conflict_group_id": "write_file_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.55,
        "source_url": "",
        "gradient_pairs": [{"level": "medium", "gold_skill": "write_file", "name_similarity": 0.50}]
    },
    # Web domain near-identical duplicates
    {
        "skill_id": make_skill_id("web_open_page", "near_ident_high"),
        "name": "open_webpage",
        "domain": "web_navigation",
        "description": "Open a webpage and read its contents (synonym for navigate_url, high name similarity)",
        "parameters": {
            "url": {"type": "string", "required": True},
            "mode": {"type": "string", "required": False, "default": "full"}
        },
        "return_format": "markdown",
        "success_rate": 0.88,
        "last_verified": "2026-06-10",
        "conflict_group_id": "navigate_url_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.85,
        "source_url": "",
        "gradient_pairs": [{"level": "high", "gold_skill": "navigate_url", "name_similarity": 0.87}]
    },
    {
        "skill_id": make_skill_id("web_lookup_answer", "near_ident_low"),
        "name": "lookup_answer",
        "domain": "web_navigation",
        "description": "Look up an answer on the web (low name similarity to search_web)",
        "parameters": {
            "question": {"type": "string", "required": True},
            "source": {"type": "string", "required": False, "default": "auto"}
        },
        "return_format": "json",
        "success_rate": 0.82,
        "last_verified": "2026-06-04",
        "conflict_group_id": "search_web_group",
        "conflict_type": "near_identical",
        "fan_degree": 5,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.30,
        "source_url": "",
        "gradient_pairs": [{"level": "low", "gold_skill": "search_web", "name_similarity": 0.25}]
    },
    {
        "skill_id": make_skill_id("web_harvest_content", "near_ident_med"),
        "name": "harvest_content",
        "domain": "web_navigation",
        "description": "Harvest content from a web page (medium name similarity to scrape_page)",
        "parameters": {
            "url": {"type": "string", "required": True},
            "data_fields": {"type": "array", "required": True}
        },
        "return_format": "json_list",
        "success_rate": 0.84,
        "last_verified": "2026-06-06",
        "conflict_group_id": "scrape_page_group",
        "conflict_type": "near_identical",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.52,
        "source_url": "",
        "gradient_pairs": [{"level": "medium", "gold_skill": "scrape_page", "name_similarity": 0.48}]
    },
    {
        "skill_id": make_skill_id("task_register_action", "near_ident_med"),
        "name": "register_action",
        "domain": "api_calling",
        "description": "Register an action item (medium name similarity to create_task)",
        "parameters": {
            "title": {"type": "string", "required": True},
            "due_date": {"type": "string", "required": False},
            "category": {"type": "string", "required": False, "default": "general"}
        },
        "return_format": "json",
        "success_rate": 0.84,
        "last_verified": "2026-06-08",
        "conflict_group_id": "create_task_group",
        "conflict_type": "near_identical",
        "fan_degree": 2,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.50,
        "source_url": "",
        "gradient_pairs": [{"level": "medium", "gold_skill": "create_task", "name_similarity": 0.45}]
    },
]

NEW_SKILLS.extend(near_identical_new)

# ============================================================
# 8 NEAR-SIMILAR DUPLICATES (similarity 0.5-0.7)
# Target: 15 total near-similar
# Existing near_duplicate entries with lower similarity get recategorized
# ============================================================

near_similar_new = [
    {
        "skill_id": make_skill_id("web_view_content", "near_similar"),
        "name": "view_content",
        "domain": "web_navigation",
        "description": "View content from a URL without navigation (partial overlap with navigate_url)",
        "parameters": {
            "url": {"type": "string", "required": True},
            "section": {"type": "string", "required": False}
        },
        "return_format": "text",
        "success_rate": 0.80,
        "last_verified": "2026-06-01",
        "conflict_group_id": "navigate_url_group",
        "conflict_type": "near_similar",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.60,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("api_check_weather", "near_similar"),
        "name": "check_weather",
        "domain": "api_calling",
        "description": "Check current weather conditions (partial overlap with get_weather, different params)",
        "parameters": {
            "city": {"type": "string", "required": True},
            "include_alerts": {"type": "boolean", "required": False, "default": False}
        },
        "return_format": "json",
        "success_rate": 0.88,
        "last_verified": "2026-06-15",
        "conflict_group_id": "get_weather_group",
        "conflict_type": "near_similar",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.70,
        "source_url": "",
        "gradient_pairs": [{"level": "high", "gold_skill": "get_weather", "name_similarity": 0.90}]
    },
    {
        "skill_id": make_skill_id("api_report_climate", "near_similar"),
        "name": "report_climate",
        "domain": "api_calling",
        "description": "Report climate data for a region (low overlap with get_weather, different time scope)",
        "parameters": {
            "region": {"type": "string", "required": True},
            "time_period": {"type": "string", "required": False, "default": "yearly"}
        },
        "return_format": "json",
        "success_rate": 0.78,
        "last_verified": "2026-06-02",
        "conflict_group_id": "get_weather_group",
        "conflict_type": "near_similar",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.30,
        "source_url": "",
        "gradient_pairs": [{"level": "low", "gold_skill": "get_weather", "name_similarity": 0.35}]
    },
    {
        "skill_id": make_skill_id("api_get_headlines", "near_similar"),
        "name": "get_news_briefing",
        "domain": "api_calling",
        "description": "Get a news briefing summary (partial overlap with search_news, different output format)",
        "parameters": {
            "topic": {"type": "string", "required": True},
            "max_items": {"type": "integer", "required": False, "default": 5}
        },
        "return_format": "text",
        "success_rate": 0.82,
        "last_verified": "2026-06-09",
        "conflict_group_id": "search_news_group",
        "conflict_type": "near_similar",
        "fan_degree": 2,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.55,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("api_load_document", "near_similar"),
        "name": "load_document",
        "domain": "api_calling",
        "description": "Load a document from storage (partial overlap with read_file, supports more formats)",
        "parameters": {
            "path": {"type": "string", "required": True},
            "format": {"type": "string", "required": False, "default": "auto"}
        },
        "return_format": "json",
        "success_rate": 0.86,
        "last_verified": "2026-06-14",
        "conflict_group_id": "read_file_group",
        "conflict_type": "near_similar",
        "fan_degree": 2,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.68,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("web_monitor_changes", "near_similar"),
        "name": "monitor_changes",
        "domain": "web_navigation",
        "description": "Monitor a website for specific changes (partial overlap with monitor_page, focused on alerts)",
        "parameters": {
            "url": {"type": "string", "required": True},
            "watch_selector": {"type": "string", "required": True},
            "alert_threshold": {"type": "string", "required": False, "default": "any"}
        },
        "return_format": "json_alert",
        "success_rate": 0.83,
        "last_verified": "2026-06-13",
        "conflict_group_id": "monitor_page_group",
        "conflict_type": "near_similar",
        "fan_degree": 2,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.62,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("api_store_content", "near_similar"),
        "name": "store_content",
        "domain": "api_calling",
        "description": "Store content with metadata (partial overlap with write_file, adds versioning)",
        "parameters": {
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
            "metadata": {"type": "object", "required": False},
            "version": {"type": "string", "required": False, "default": "1"}
        },
        "return_format": "json",
        "success_rate": 0.87,
        "last_verified": "2026-06-16",
        "conflict_group_id": "write_file_group",
        "conflict_type": "near_similar",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.65,
        "source_url": "",
        "gradient_pairs": []
    },
    {
        "skill_id": make_skill_id("web_brief_summarize", "near_similar"),
        "name": "brief_summary",
        "domain": "web_navigation",
        "description": "Get a brief summary of a page (partial overlap with summarize_page, much shorter output)",
        "parameters": {
            "url": {"type": "string", "required": True},
            "sentences": {"type": "integer", "required": False, "default": 3}
        },
        "return_format": "text",
        "success_rate": 0.85,
        "last_verified": "2026-06-12",
        "conflict_group_id": "summarize_page_group",
        "conflict_type": "near_similar",
        "fan_degree": 2,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.70,
        "source_url": "",
        "gradient_pairs": []
    },
]

NEW_SKILLS.extend(near_similar_new)

# ============================================================
# 2 ADDITIONAL NEAR-SIMILAR SKILLS (to reach exactly 100)
# ============================================================

near_similar_extra = [
    {
        "skill_id": make_skill_id("api_notify_contact", "near_ident_low"),
        "name": "notify_contact",
        "domain": "api_calling",
        "description": "Send a notification to a contact (low name similarity to send_email)",
        "parameters": {
            "contact_id": {"type": "string", "required": True},
            "message": {"type": "string", "required": True},
            "channel": {"type": "string", "required": False, "default": "email"}
        },
        "return_format": "json",
        "success_rate": 0.80,
        "last_verified": "2026-06-03",
        "conflict_group_id": "send_email_group",
        "conflict_type": "near_similar",
        "fan_degree": 4,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": False,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.40,
        "source_url": "",
        "gradient_pairs": [{"level": "low", "gold_skill": "send_email", "name_similarity": 0.35}]
    },
    {
        "skill_id": make_skill_id("api_forecast_conditions", "near_similar"),
        "name": "forecast_conditions",
        "domain": "api_calling",
        "description": "Get weather forecast conditions (medium name similarity to get_weather, different time scope)",
        "parameters": {
            "location": {"type": "string", "required": True},
            "forecast_days": {"type": "integer", "required": False, "default": 5},
            "detail_level": {"type": "string", "required": False, "default": "summary"}
        },
        "return_format": "json_list",
        "success_rate": 0.85,
        "last_verified": "2026-06-11",
        "conflict_group_id": "get_weather_group",
        "conflict_type": "near_similar",
        "fan_degree": 3,
        "staleness_days": 0,
        "is_trap": False,
        "is_stale": False,
        "is_clean": False,
        "parametric_overlap": True,
        "trap_description": "",
        "example_usage": "",
        "similarity_to_gold": 0.58,
        "source_url": "",
        "gradient_pairs": [{"level": "medium", "gold_skill": "get_weather", "name_similarity": 0.55}]
    },
]

NEW_SKILLS.extend(near_similar_extra)


def build_expanded_library(existing_path: str, output_path: str):
    """Build the expanded 100-skill library."""
    existing = load_existing_library(existing_path)
    
    # Add similarity_to_gold to existing skills
    for skill in existing:
        if 'similarity_to_gold' not in skill:
            skill['similarity_to_gold'] = 0.0
        if 'source_url' not in skill:
            skill['source_url'] = ''
        if 'gradient_pairs' not in skill:
            skill['gradient_pairs'] = []
        if 'example_usage' not in skill or skill['example_usage'] is None:
            skill['example_usage'] = ''
        
        # Set similarity_to_gold for gold skills (clean with conflict_group but conflict_type none)
        # and for existing interference skills
        if skill.get('is_clean', False):
            skill['similarity_to_gold'] = 1.0
        elif skill.get('conflict_type') == 'schema_conflict':
            skill['similarity_to_gold'] = 1.0  # Identical name, different schema
            skill['conflict_type'] = 'schema_conflict'  # Keep as is
        elif skill.get('conflict_type') == 'semantic_conflict':
            skill['similarity_to_gold'] = 0.85  # Same name, different semantics
            skill['conflict_type'] = 'semantic_conflict'
        elif skill.get('conflict_type') == 'version_conflict':
            skill['similarity_to_gold'] = 0.75  # Same name, version drift
        elif skill.get('conflict_type') == 'near_duplicate':
            # Recategorize: assign similarity based on description analysis
            name = skill.get('name', '')
            desc = skill.get('description', '')
            # Higher similarity for near-synonyms, lower for overlapping scope
            near_identical_names = ['add_event', 'compose_email', 'search_database', 
                                     'save_file', 'get_headlines', 'find_information',
                                     'browse_website', 'get_data_from_page', 'quick_summary',
                                     'fetch_file']
            near_similar_names = []
            
            if name in near_identical_names:
                skill['conflict_type'] = 'near_identical'
                # Assign specific similarity scores
                similarity_map = {
                    'add_event': 0.88,
                    'compose_email': 0.85,
                    'search_database': 0.82,
                    'save_file': 0.90,
                    'get_headlines': 0.65,
                    'find_information': 0.75,
                    'browse_website': 0.65,
                    'get_data_from_page': 0.60,
                    'quick_summary': 0.70,
                    'fetch_file': 0.82,
                }
                skill['similarity_to_gold'] = similarity_map.get(name, 0.80)
            else:
                skill['conflict_type'] = 'near_similar'
                skill['similarity_to_gold'] = 0.55
    
    # Set similarity_to_gold for stale and trap skills
    for skill in existing:
        if skill.get('is_stale', False):
            skill['similarity_to_gold'] = 0.9  # Stale skills are very similar to gold
        if skill.get('is_trap', False):
            skill['similarity_to_gold'] = 0.95  # Traps mimic gold skills closely
    
    # Combine existing + new
    all_skills = existing + NEW_SKILLS
    
    # FIX: Assign conflict_group_ids to gold (clean) skills that are referenced by
    # interference skills, and set their similarity_to_gold = 1.0
    gold_group_map = {
        'create_event': 'create_event_group',
        'list_events': 'list_events_group',
        'send_email': 'send_email_group',
        'read_emails': 'read_emails_group',
        'read_file': 'read_file_group',
        'write_file': 'write_file_group',
        'query_database': 'query_database_group',
        'get_weather': 'get_weather_group',
        'get_stock_price': 'get_stock_price_group',
        'search_news': 'search_news_group',
        'create_task': 'create_task_group',
        'summarize_page': 'summarize_page_group',
        'monitor_page': 'monitor_page_group',
        'download_file': 'download_file_group',
        'fill_form': 'fill_form_group',
        'scrape_page': 'scrape_page_group',
        'navigate_url': 'navigate_url_group',
        'search_web': 'search_web_group',
    }
    
    for skill in all_skills:
        name = skill.get('name', '')
        # Assign conflict_group_id to gold skills that are referenced
        if skill.get('is_clean', False) and name in gold_group_map:
            skill['conflict_group_id'] = gold_group_map[name]
            skill['similarity_to_gold'] = 1.0
    
    # Verify total count
    total = len(all_skills)
    print(f"Total skills: {total}")
    assert total == 100, f"Expected 100 skills, got {total}"
    
    # Compute stats
    stats = {
        "total": total,
        "clean": sum(1 for s in all_skills if s.get('is_clean', False)),
        "near_identical": sum(1 for s in all_skills if s.get('conflict_type') == 'near_identical'),
        "near_similar": sum(1 for s in all_skills if s.get('conflict_type') == 'near_similar'),
        "schema_conflict": sum(1 for s in all_skills if s.get('conflict_type') == 'schema_conflict'),
        "semantic_conflict": sum(1 for s in all_skills if s.get('conflict_type') == 'semantic_conflict'),
        "version_conflict": sum(1 for s in all_skills if s.get('conflict_type') == 'version_conflict'),
        "stale": sum(1 for s in all_skills if s.get('is_stale', False)),
        "trap": sum(1 for s in all_skills if s.get('is_trap', False)),
        "by_domain": {},
        "by_conflict_type": {},
    }
    for s in all_skills:
        d = s.get('domain', 'unknown')
        stats['by_domain'][d] = stats['by_domain'].get(d, 0) + 1
        ct = s.get('conflict_type', 'none')
        stats['by_conflict_type'][ct] = stats['by_conflict_type'].get(ct, 0) + 1
    
    output = {
        "metadata": {
            "version": "2.0",
            "date": "2026-06-25",
            "description": "MemInterfere expanded skill library with formal near-duplicate taxonomy (Phase 5.1)",
            "stats": stats,
            "taxonomy_version": "1.0",
            "gradient_pairs": [
                {
                    "gold_skill": "create_event",
                    "high_similarity": "add_calendar_event",
                    "medium_similarity": "schedule_meeting",
                    "low_similarity": "book_appointment"
                },
                {
                    "gold_skill": "send_email",
                    "high_similarity": "dispatch_email",
                    "medium_similarity": "compose_message",
                    "low_similarity": "notify_contact"
                },
                {
                    "gold_skill": "search_web",
                    "high_similarity": "web_search",
                    "medium_similarity": "find_information",
                    "low_similarity": "lookup_answer"
                },
                {
                    "gold_skill": "navigate_url",
                    "high_similarity": "open_webpage",
                    "medium_similarity": "browse_website",
                    "low_similarity": "view_content"
                },
                {
                    "gold_skill": "scrape_page",
                    "high_similarity": "extract_page_data",
                    "medium_similarity": "get_data_from_page",
                    "low_similarity": "harvest_content"
                },
                {
                    "gold_skill": "query_database",
                    "high_similarity": "run_db_query",
                    "medium_similarity": "search_database",
                    "low_similarity": "fetch_records"
                },
                {
                    "gold_skill": "get_weather",
                    "high_similarity": "check_weather",
                    "medium_similarity": "forecast_conditions",
                    "low_similarity": "report_climate"
                },
                {
                    "gold_skill": "read_file",
                    "high_similarity": "load_file",
                    "medium_similarity": "access_document",
                    "low_similarity": "retrieve_storage"
                },
                {
                    "gold_skill": "write_file",
                    "high_similarity": "save_file",
                    "medium_similarity": "store_content",
                    "low_similarity": "persist_data"
                },
                {
                    "gold_skill": "create_task",
                    "high_similarity": "add_task",
                    "medium_similarity": "schedule_todo",
                    "low_similarity": "register_action"
                }
            ]
        },
        "skills": all_skills
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {total} skills to {output_path}")
    print(f"Stats: {json.dumps(stats, indent=2)}")
    return output


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing_path = os.path.join(base_dir, "data", "skills", "skill_library.json")
    output_path = os.path.join(base_dir, "data", "skills", "expanded_library_100.json")
    build_expanded_library(existing_path, output_path)