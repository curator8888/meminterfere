"""
MemInterfere: Expanded Task Set (Phase 3)

Adds 33 new tasks including:
- 15 natural interference tasks (interference emerges organically from task phrasing)
- 10 hard multi-step composition tasks
- 8 medium difficulty tasks

Total: 47 + 33 = 80 tasks
"""

from metrics import EvalTask, EVAL_TASKS

# ============================================================
# NATURAL INTERFERENCE TASKS (15)
# These are tasks where interference occurs ORGANICALLY — the task
# phrasing doesn't mention specific skills, but multiple skills 
# could plausibly handle it. The agent must choose correctly.
# ============================================================

natural_interference_tasks = [
    # Web navigation natural interference
    EvalTask(task_id="nat_web_001", 
             description="I need to find out what the weather will be like in Tokyo next week for a business trip",
             domain="web_navigation", difficulty="easy", 
             expected_skill_ids=["search_web"],
             expected_outcome="Weather forecast information for Tokyo",
             interference_potential="high",  # could trigger search_web, navigate_url, or scrape_page
             requires_memory=False, natural_interference=True),
    
    EvalTask(task_id="nat_web_002",
             description="Get me the latest headlines about AI regulation from a news site",
             domain="web_navigation", difficulty="easy",
             expected_skill_ids=["search_web"],  # or read_rss, scrape_page, extract_links
             expected_outcome="List of AI regulation headlines",
             interference_potential="high",
             requires_memory=False, natural_interference=True),
    
    EvalTask(task_id="nat_web_003",
             description="I want to save a copy of the report at https://reports.example.com/annual-2026.pdf",
             domain="web_navigation", difficulty="easy",
             expected_skill_ids=["download_file"],  # or scrape_page, navigate_url
             expected_outcome="Downloaded PDF file path",
             interference_potential="medium",
             requires_memory=False, natural_interference=True),
    
    EvalTask(task_id="nat_web_004",
             description="Can you tell me what's on the page at https://docs.example.com/api/v2 without downloading anything?",
             domain="web_navigation", difficulty="easy",
             expected_skill_ids=["navigate_url"],  # or scrape_page, extract_metadata
             expected_outcome="Page content in markdown format",
             interference_potential="medium",
             requires_memory=False, natural_interference=True),
    
    EvalTask(task_id="nat_web_005",
             description="Find all the links on https://example.com that point to PDF files and download them",
             domain="web_navigation", difficulty="medium",
             expected_skill_ids=["extract_links", "download_file"],
             expected_outcome="List of downloaded PDFs",
             interference_potential="high",  # multi-step, could trigger many skills
             requires_memory=True, natural_interference=True),
    
    EvalTask(task_id="nat_web_006",
             description="Keep an eye on https://status.example.com and let me know when it changes",
             domain="web_navigation", difficulty="medium",
             expected_skill_ids=["monitor_page"],  # or scrape_page + check_website_status
             expected_outcome="Monitoring setup with change detection",
             interference_potential="high",
             requires_memory=True, natural_interference=True),
    
    EvalTask(task_id="nat_web_007",
             description="I need a quick overview of what https://arxiv.org/abs/2401.12345 is about",
             domain="web_navigation", difficulty="easy",
             expected_skill_ids=["summarize_page"],  # or navigate_url, extract_metadata, quick_summary
             expected_outcome="Brief summary of the paper",
             interference_potential="high",  # summarize_page vs quick_summary vs navigate_url
             requires_memory=False, natural_interference=True),
    
    # API calling natural interference
    EvalTask(task_id="nat_api_001",
             description="Set up a meeting with the team for next Thursday at 3pm and remind me 30 minutes before",
             domain="api_calling", difficulty="medium",
             expected_skill_ids=["create_event", "set_reminder"],
             expected_outcome="Calendar event created + reminder set",
             interference_potential="high",  # create_event vs add_event, and reminder selection
             requires_memory=True, natural_interference=True),
    
    EvalTask(task_id="nat_api_002",
             description="Find out what John's email address is and send him the project update",
             domain="api_calling", difficulty="medium",
             expected_skill_ids=["lookup_contact", "send_email"],
             expected_outcome="Email sent to John with project update",
             interference_potential="high",  # send_email vs compose_email
             requires_memory=True, natural_interference=True),
    
    EvalTask(task_id="nat_api_003",
             description="Show me all the tasks I need to do this week",
             domain="api_calling", difficulty="easy",
             expected_skill_ids=["list_tasks"],  # or read_emails (if tasks come by email)
             expected_outcome="JSON list of open tasks",
             interference_potential="medium",
             requires_memory=False, natural_interference=True),
    
    EvalTask(task_id="nat_api_004",
             description="What's the weather like in San Francisco? I need to decide whether to schedule a hike this weekend.",
             domain="api_calling", difficulty="easy",
             expected_skill_ids=["get_weather"],
             expected_outcome="Weather data for San Francisco",
             interference_potential="medium",  # get_weather has semantic and stale conflicts
             requires_memory=False, natural_interference=True),
    
    EvalTask(task_id="nat_api_005",
             description="Read the email from Sarah about the budget and save the key numbers to a file",
             domain="api_calling", difficulty="hard",
             expected_skill_ids=["read_emails", "write_file"],
             expected_outcome="File with budget numbers extracted from Sarah's email",
             interference_potential="high",  # read_emails has conflicts, write_file has version conflicts
             requires_memory=True, natural_interference=True),
    
    EvalTask(task_id="nat_api_006",
             description="Look up the current Apple stock price and tell me if it's higher than yesterday",
             domain="api_calling", difficulty="medium",
             expected_skill_ids=["get_stock_price", "calculate"],
             expected_outcome="Comparison of current vs yesterday's AAPL price",
             interference_potential="high",  # get_stock_price has semantic and stale conflicts
             requires_memory=True, natural_interference=True),
    
    EvalTask(task_id="nat_api_007",
             description="Create a to-do item for reviewing the quarterly report and another for booking the conference room",
             domain="api_calling", difficulty="easy",
             expected_skill_ids=["create_task"],
             expected_outcome="Two tasks created",
             interference_potential="high",  # create_task has version and near-duplicate conflicts
             requires_memory=False, natural_interference=True),
    
    EvalTask(task_id="nat_api_008",
             description="Run a database query to find all customers who signed up last month and email the results to the marketing team",
             domain="api_calling", difficulty="hard",
             expected_skill_ids=["query_database", "write_file", "send_email"],
             expected_outcome="Customer list emailed to marketing team",
             interference_potential="high",  # query_database has schema+version+nd conflicts, send_email has conflicts
             requires_memory=True, natural_interference=True),
]

# ============================================================
# ADDITIONAL HARD TASKS (10) — Multi-step composition
# These require composing 3+ skills correctly
# ============================================================

additional_hard_tasks = [
    EvalTask(task_id="hard_web_001",
             description="Search for 'machine learning conferences 2026', scrape the top 5 results for event details, extract their dates and locations, and create a comparison table",
             domain="web_navigation", difficulty="hard",
             expected_skill_ids=["search_web", "scrape_page", "extract_table"],
             expected_outcome="Comparison table of ML conferences with dates and locations",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_web_002",
             description="Crawl https://docs.example.com for all pages about authentication, extract their metadata, and download the main PDF guide",
             domain="web_navigation", difficulty="hard",
             expected_skill_ids=["crawl_site", "extract_metadata", "download_file"],
             expected_outcome="Downloaded PDF + metadata for authentication docs",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_web_003",
             description="Monitor 3 competitor websites daily, extract any pricing changes, summarize the differences, and save a report",
             domain="web_navigation", difficulty="hard",
             expected_skill_ids=["monitor_page", "scrape_page", "extract_table", "summarize_page", "write_file"],
             expected_outcome="Daily competitive pricing report saved to file",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_web_004",
             description="Read the RSS feed from 2 tech blogs, find articles about Rust programming, navigate to each article, and summarize the top 3",
             domain="web_navigation", difficulty="hard",
             expected_skill_ids=["read_rss", "navigate_url", "summarize_page"],
             expected_outcome="Summaries of top 3 Rust articles from 2 blogs",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_web_005",
             description="Check the status of 5 microservices, take screenshots of the ones that are down, and scrape their error pages for details",
             domain="web_navigation", difficulty="hard",
             expected_skill_ids=["check_website_status", "take_screenshot", "scrape_page"],
             expected_outcome="Status report with screenshots and error details for down services",
             interference_potential="medium",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_api_001",
             description="Read all unread emails, identify meeting invites, create calendar events for each, set reminders 15 minutes before, and reply confirming attendance",
             domain="api_calling", difficulty="hard",
             expected_skill_ids=["read_emails", "create_event", "set_reminder", "send_email"],
             expected_outcome="Calendar events created with reminders and confirmation emails sent",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_api_002",
             description="Query the database for Q1 revenue by region, save results to a CSV file, look up the VP of Sales contact, and email them the file",
             domain="api_calling", difficulty="hard",
             expected_skill_ids=["query_database", "write_file", "lookup_contact", "send_email"],
             expected_outcome="Q1 revenue CSV emailed to VP of Sales",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_api_003",
             description="Get the weather forecast for 3 cities where we have offices, create calendar events for outdoor team activities on the best weather day, and send invites to all team members",
             domain="api_calling", difficulty="hard",
             expected_skill_ids=["get_weather", "create_event", "lookup_contact", "send_email"],
             expected_outcome="Weather-informed outdoor events created and invites sent",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_api_004",
             description="Pull this month's stock prices for our portfolio companies, calculate the percentage change, write a summary report, and create follow-up tasks for the analysts",
             domain="api_calling", difficulty="hard",
             expected_skill_ids=["get_stock_price", "calculate", "write_file", "create_task"],
             expected_outcome="Portfolio performance report with analyst tasks",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="hard_api_005",
             description="Search news about data privacy regulation changes, create tasks for each relevant article, set reminders for compliance deadlines, and email the legal team",
             domain="api_calling", difficulty="hard",
             expected_skill_ids=["search_news", "create_task", "set_reminder", "send_email"],
             expected_outcome="Privacy regulation tasks with reminders and legal team notification",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
]

# ============================================================
# ADDITIONAL MEDIUM TASKS (8) — Requiring memory with moderate interference
# ============================================================

additional_medium_tasks = [
    EvalTask(task_id="med_web_001",
             description="Extract the price table from https://shop.example.com/products and save it as a structured data file",
             domain="web_navigation", difficulty="medium",
             expected_skill_ids=["extract_table", "write_file"],
             expected_outcome="Structured product pricing data saved to file",
             interference_potential="medium",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="med_web_002",
             description="Find and summarize 3 recent articles about climate policy from https://news.example.com",
             domain="web_navigation", difficulty="medium",
             expected_skill_ids=["search_web", "navigate_url", "summarize_page"],
             expected_outcome="Summaries of 3 climate policy articles",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="med_web_003",
             description="Navigate to https://github.com/example/repo/issues and extract the titles and labels of all open issues",
             domain="web_navigation", difficulty="medium",
             expected_skill_ids=["navigate_url", "scrape_page"],
             expected_outcome="JSON list of issue titles and labels",
             interference_potential="medium",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="med_web_004",
             description="Download the dataset from https://data.example.com/demographics.csv and verify it's readable",
             domain="web_navigation", difficulty="medium",
             expected_skill_ids=["download_file", "read_file"],
             expected_outcome="Verified downloaded CSV file",
             interference_potential="medium",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="med_api_001",
             description="Look up the contact info for 'David Chen' and send them a meeting invitation for next Monday at 10am",
             domain="api_calling", difficulty="medium",
             expected_skill_ids=["lookup_contact", "send_email"],
             expected_outcome="Meeting invitation sent to David Chen",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="med_api_002",
             description="Calculate the total revenue from this list: [150000, 230000, 185000, 210000] and create a task to review it",
             domain="api_calling", difficulty="medium",
             expected_skill_ids=["calculate", "create_task"],
             expected_outcome="Sum of revenues + task created for review",
             interference_potential="medium",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="med_api_003",
             description="Read the last 5 emails, find the one from the boss, and create a task based on its content",
             domain="api_calling", difficulty="medium",
             expected_skill_ids=["read_emails", "create_task"],
             expected_outcome="Task created from boss's email",
             interference_potential="high",
             requires_memory=True, natural_interference=False),
    
    EvalTask(task_id="med_api_004",
             description="Get the weather for New York, and if it's going to rain, set a reminder to bring an umbrella tomorrow morning",
             domain="api_calling", difficulty="medium",
             expected_skill_ids=["get_weather", "set_reminder"],
             expected_outcome="Conditional reminder set based on weather",
             interference_potential="medium",
             requires_memory=True, natural_interference=False),
]

# Combine all tasks
ALL_TASKS = EVAL_TASKS + natural_interference_tasks + additional_hard_tasks + additional_medium_tasks

if __name__ == "__main__":
    import json, os
    from collections import Counter
    
    total = len(ALL_TASKS)
    diff_counts = Counter(t.difficulty for t in ALL_TASKS)
    domain_counts = Counter(t.domain for t in ALL_TASKS)
    interf_counts = Counter(t.interference_potential for t in ALL_TASKS)
    mem_counts = Counter('yes' if t.requires_memory else 'no' for t in ALL_TASKS)
    natural_counts = Counter('yes' if t.natural_interference else 'no' for t in ALL_TASKS)
    
    print(f"Total tasks: {total}")
    print(f"\nBy difficulty: {dict(diff_counts)}")
    print(f"By domain: {dict(domain_counts)}")
    print(f"By interference: {dict(interf_counts)}")
    print(f"Requires memory: {dict(mem_counts)}")
    print(f"Natural interference: {dict(natural_counts)}")
    
    # Save
    os.makedirs("data/tasks", exist_ok=True)
    tasks_data = {
        "metadata": {
            "version": "3.0",
            "date": "2026-06-25",
            "description": "MemInterfere evaluation tasks v3 - expanded with natural interference",
            "stats": {
                "total": total,
                "by_difficulty": dict(diff_counts),
                "by_domain": dict(domain_counts),
                "by_interference": dict(interf_counts),
                "requires_memory": dict(mem_counts),
                "natural_interference": dict(natural_counts),
            }
        },
        "tasks": [t.__dict__ if hasattr(t, '__dict__') else {
            'task_id': t.task_id, 'description': t.description, 'domain': t.domain,
            'difficulty': t.difficulty, 'expected_skill_ids': t.expected_skill_ids,
            'expected_outcome': t.expected_outcome, 'interference_potential': t.interference_potential,
            'requires_memory': t.requires_memory, 'natural_interference': t.natural_interference
        } for t in ALL_TASKS]
    }
    
    with open("data/tasks/eval_tasks.json", "w") as f:
        json.dump(tasks_data, f, indent=2)
    
    print(f"\nSaved {total} tasks to data/tasks/eval_tasks.json")