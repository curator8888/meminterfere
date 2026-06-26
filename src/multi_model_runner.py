"""
MemInterfere: Multi-Model Evaluation Runner

Runs evaluation tasks across multiple LLM backends with robust response parsing,
rate limiting, retry logic, and incremental result saving.

Model tiers:
  - Small:    Llama-3.1-8b-instruct  (OpenRouter, free)
  - Medium:   Grok-3-mini            (xAI, $0.30/M)
  - Capable:  GPT-4o-mini            (OpenRouter, $0.15/M)
  - Strong:   Claude-3.5-Haiku        (Anthropic, $0.25/M)
"""

import json
import os
import sys
import time
import asyncio
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from model_config import ModelConfig, MODELS, get_model, estimate_cost
from response_parser import parse_response, ParsedResponse, ParseStats, PARSE_METHOD_ERROR

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Complete result from a single model run."""
    # Identity
    run_id: str
    model: str
    model_id: str
    task_id: str
    condition: str
    temperature: float

    # Timing
    latency_ms: int
    timestamp: str

    # Token usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Input/output
    prompt_text: str
    raw_response: str

    # Parsed result
    parsed_tool: str
    parsed_confidence: float
    parsed_approach: str
    parse_method: str
    all_matched_skills: list[str] = field(default_factory=list)

    # Ground truth
    expected_skills: list[str] = field(default_factory=list)
    skill_correct: bool = False  # Whether parsed_tool matches expected

    # Error info
    error: str = ""  # Non-empty if API call failed after retries


@dataclass
class RateLimiter:
    """Simple rate limiter that tracks requests per minute per provider."""
    provider: str
    max_rpm: int
    _timestamps: list[float] = field(default_factory=list)

    def wait_if_needed(self):
        """Block until we can make another request within the RPM limit."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max_rpm:
            # Wait until the oldest request ages out
            sleep_time = 60 - (now - self._timestamps[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit for {self.provider}: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._timestamps.append(time.time())


# ── Prompt construction ──────────────────────────────────────────────────────

def build_system_prompt(skill_library: list, condition: str) -> str:
    """Build the system prompt including the skill library."""
    parts = [
        "You are an AI assistant with access to a skill library. Given a task, you must select",
        "the most appropriate skill from the library and describe how you would use it.",
        "",
        "## Skill Library",
        "",
    ]

    for skill in skill_library:
        # Format skill entry
        params_str = ", ".join(
            f"{k}: {v.get('type', 'any')}"
            for k, v in (skill.parameters if isinstance(skill.parameters, dict) else {}).items()
        )
        staleness_note = " [STALE]" if getattr(skill, 'is_stale', False) else ""
        trap_note = " [WARNING: SUSPICIOUS]" if getattr(skill, 'is_trap', False) else ""
        conflict_note = f" [CONFLICT: {skill.conflict_type}]" if skill.conflict_type != "none" else ""

        parts.append(
            f"- **{skill.name}** (id: {skill.skill_id}): {skill.description}"
            f"{staleness_note}{trap_note}{conflict_note}"
            f"\n  Parameters: {params_str}"
        )

    parts.extend([
        "",
        "## Instructions",
        "",
        "Given a task description, respond with EXACTLY this format:",
        "",
        "TOOL: <skill_name>",
        "CONFIDENCE: <number between 0 and 1>",
        "APPROACH: <brief description of how you would use the tool>",
        "",
        "Alternatively, you can respond in JSON format:",
        '```json',
        '{"tool": "<skill_name>", "confidence": <number>, "approach": "<description>"}',
        '```',
        "",
        "IMPORTANT: Choose the most appropriate skill from the library above. "
        "Be careful with skills marked [STALE], [WARNING], or [CONFLICT].",
    ])

    return "\n".join(parts)


def build_user_prompt(task) -> str:
    """Build the user prompt for a single task."""
    return (
        f"Task: {task.description}\n\n"
        f"Expected outcome: {task.expected_outcome}\n\n"
        f"Which skill from the library would you use for this task, "
        f"and how would you approach it?"
    )


def build_messages_for_anthropic(system_prompt: str, user_prompt: str) -> list[dict]:
    """Build the message list for Anthropic API (which doesn't use system in messages)."""
    return [
        {"role": "user", "content": user_prompt},
    ]


def build_messages_for_openai(system_prompt: str, user_prompt: str) -> list[dict]:
    """Build the message list for OpenAI-compatible APIs."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ── API call functions ───────────────────────────────────────────────────────

def call_openai_compatible(
    config: ModelConfig,
    messages: list[dict],
    temperature: float,
    max_retries: int = 3,
) -> tuple[str, int, int, int]:
    """
    Call an OpenAI-compatible API (xAI, OpenRouter).
    Returns (response_text, prompt_tokens, completion_tokens, total_tokens).
    """
    api_key = os.environ.get(config.env_key, "")
    if not api_key:
        raise ValueError(f"Missing API key: {config.env_key}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(config.extra_headers)

    payload = {
        "model": config.model_id,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": temperature,
    }

    # Add JSON mode if supported
    if config.supports_json_mode:
        payload["response_format"] = {"type": "json_object"}

    url = f"{config.api_base}/chat/completions"

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=headers, json=payload)

            if resp.status_code == 429:
                # Rate limited
                retry_after = int(resp.headers.get("retry-after", "60"))
                wait_time = 2 ** attempt + retry_after
                logger.warning(f"Rate limited (429) for {config.name}, waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            resp.raise_for_status()
            data = resp.json()

            response_text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return response_text, prompt_tokens, completion_tokens, total_tokens

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error for {config.name} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                time.sleep(wait_time)
            else:
                raise

        except Exception as e:
            logger.warning(f"Error calling {config.name} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed to call {config.name} after {max_retries} retries")


def call_anthropic(
    config: ModelConfig,
    system_prompt: str,
    messages: list[dict],
    temperature: float,
    max_retries: int = 3,
) -> tuple[str, int, int, int]:
    """
    Call the Anthropic Messages API.
    Returns (response_text, input_tokens, output_tokens, total_tokens).
    """
    api_key = os.environ.get(config.env_key, "")
    if not api_key:
        raise ValueError(f"Missing API key: {config.env_key}")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Anthropic uses a separate system parameter
    # Note: Anthropic requires temperature > 0; if 0.0 is passed, omit it (defaults to 1.0)
    # For reproducibility with low randomness, use 0.01 instead of 0.0
    effective_temp = max(0.01, temperature) if temperature <= 0.0 else temperature
    payload = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "temperature": effective_temp,
        "system": system_prompt,
        "messages": messages,
    }

    url = f"{config.api_base}/messages"

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=headers, json=payload)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", "30"))
                wait_time = 2 ** attempt + retry_after
                logger.warning(f"Rate limited (429) for {config.name}, waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Anthropic response format
            response_text = data["content"][0]["text"]
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens

            return response_text, input_tokens, output_tokens, total_tokens

        except httpx.HTTPStatusError as e:
            # Log response body for debugging (Anthropic returns detailed errors)
            try:
                error_body = e.response.text[:500]
                logger.warning(f"Anthropic error body for {config.name}: {error_body}")
            except Exception:
                pass
            logger.warning(f"HTTP error for {config.name} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                time.sleep(wait_time)
            else:
                raise

        except Exception as e:
            logger.warning(f"Error calling {config.name} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed to call {config.name} after {max_retries} retries")


def call_model(
    config: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int = 3,
) -> tuple[str, int, int, int]:
    """
    Call a model based on its provider. Returns (response_text, prompt_tokens, completion_tokens, total_tokens).
    """
    if config.provider == "anthropic":
        messages = build_messages_for_anthropic(system_prompt, user_prompt)
        return call_anthropic(config, system_prompt, messages, temperature, max_retries)
    else:
        # OpenAI-compatible (xAI, OpenRouter)
        messages = build_messages_for_openai(system_prompt, user_prompt)
        return call_openai_compatible(config, messages, temperature, max_retries)


# ── Main runner ──────────────────────────────────────────────────────────────

class MultiModelRunner:
    """Orchestrates multi-model evaluation runs with rate limiting and incremental saving."""

    def __init__(
        self,
        output_dir: str = "data/results/phase5",
        max_retries: int = 3,
    ):
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.rate_limiters: dict[str, RateLimiter] = {}
        self.results: list[RunResult] = []
        self.parse_stats = ParseStats()
        self._completed_keys: set[str] = set()

        os.makedirs(output_dir, exist_ok=True)

        # Initialize rate limiters for each provider
        for model in MODELS.values():
            if model.provider not in self.rate_limiters:
                self.rate_limiters[model.provider] = RateLimiter(
                    provider=model.provider,
                    max_rpm=model.rate_limit_rpm,
                )

    def _make_run_id(self, model: str, task_id: str, condition: str, temperature: float) -> str:
        """Create a unique run ID for deduplication."""
        raw = f"{model}|{task_id}|{condition}|{temperature}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _load_existing_results(self, results_file: str) -> int:
        """Load existing results from a file for resume capability."""
        if not os.path.exists(results_file):
            return 0

        with open(results_file) as f:
            data = json.load(f)

        if "results" not in data:
            return 0

        loaded = 0
        for r in data["results"]:
            key = f"{r['model']}|{r['task_id']}|{r['condition']}|{r['temperature']}"
            self._completed_keys.add(key)

            # Reconstruct RunResult
            result = RunResult(
                run_id=r.get("run_id", ""),
                model=r["model"],
                model_id=r.get("model_id", ""),
                task_id=r["task_id"],
                condition=r["condition"],
                temperature=r["temperature"],
                latency_ms=r.get("latency_ms", 0),
                timestamp=r.get("timestamp", ""),
                prompt_tokens=r.get("prompt_tokens", 0),
                completion_tokens=r.get("completion_tokens", 0),
                total_tokens=r.get("total_tokens", 0),
                prompt_text=r.get("prompt_text", ""),
                raw_response=r.get("raw_response", ""),
                parsed_tool=r.get("parsed_tool", ""),
                parsed_confidence=r.get("parsed_confidence", 0.0),
                parsed_approach=r.get("parsed_approach", ""),
                parse_method=r.get("parse_method", ""),
                all_matched_skills=r.get("all_matched_skills", []),
                expected_skills=r.get("expected_skills", []),
                skill_correct=r.get("skill_correct", False),
                error=r.get("error", ""),
            )
            self.results.append(result)
            self.parse_stats.record(result.parse_method)
            loaded += 1

        logger.info(f"Loaded {loaded} existing results from {results_file}")
        return loaded

    def _is_completed(self, model: str, task_id: str, condition: str, temperature: float) -> bool:
        """Check if a specific run has already been completed."""
        key = f"{model}|{task_id}|{condition}|{temperature}"
        return key in self._completed_keys

    def _save_results(self, results_file: str | None = None) -> str:
        """Save results incrementally."""
        if results_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.output_dir, f"multi_model_results_{timestamp}.json")

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_results": len(self.results),
                "parse_stats": self.parse_stats.summary(),
                "models": list(set(r.model for r in self.results)),
                "conditions": list(set(r.condition for r in self.results)),
                "temperatures": list(set(r.temperature for r in self.results)),
            },
            "results": [asdict(r) for r in self.results],
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.results)} results to {results_file}")
        return results_file

    def run_single(
        self,
        model_config: ModelConfig,
        task,
        condition_name: str,
        skill_library: list,
        temperature: float,
        valid_skill_names: list[str],
    ) -> RunResult:
        """Run a single evaluation: one model, one task, one condition, one temperature."""

        # Check for resume
        if self._is_completed(model_config.name, task.task_id, condition_name, temperature):
            # Find existing result
            for r in self.results:
                if (r.model == model_config.name and r.task_id == task.task_id
                        and r.condition == condition_name and r.temperature == temperature):
                    logger.info(f"Skipping completed: {model_config.name}/{task.task_id}/{condition_name}/T={temperature}")
                    return r

        run_id = self._make_run_id(model_config.name, task.task_id, condition_name, temperature)

        # Build prompts
        system_prompt = build_system_prompt(skill_library, condition_name)
        user_prompt = build_user_prompt(task)

        # Rate limit
        limiter = self.rate_limiters[model_config.provider]
        limiter.wait_if_needed()

        # Call model
        start_time = time.time()
        error_msg = ""
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            response_text, prompt_tokens, completion_tokens, total_tokens = call_model(
                model_config, system_prompt, user_prompt, temperature, self.max_retries
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error calling {model_config.name}: {e}")

        latency_ms = int((time.time() - start_time) * 1000)

        # Parse response
        parsed = parse_response(response_text, valid_skill_names) if response_text else ParsedResponse(
            tool_name="PARSE_ERROR",
            confidence=0.0,
            approach="",
            parse_method=PARSE_METHOD_ERROR,
            raw_response=response_text,
        )
        self.parse_stats.record(parsed.parse_method)

        # Check correctness
        expected_skills = task.expected_skill_ids if hasattr(task, 'expected_skill_ids') else []
        # Match against skill names (not IDs) since the model outputs skill names
        skill_correct = False
        if parsed.tool_name != "PARSE_ERROR":
            for expected in expected_skills:
                # Match by skill name
                for skill in skill_library:
                    if skill.skill_id == expected or skill.name == expected:
                        if parsed.tool_name == skill.name or _normalize_match(parsed.tool_name, skill.name):
                            skill_correct = True
                            break
                    if parsed.tool_name == expected:
                        skill_correct = True
                        break
                if skill_correct:
                    break

        result = RunResult(
            run_id=run_id,
            model=model_config.name,
            model_id=model_config.model_id,
            task_id=task.task_id,
            condition=condition_name,
            temperature=temperature,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_text=system_prompt[:200] + "...\n" + user_prompt,  # Truncate for storage
            raw_response=response_text,
            parsed_tool=parsed.tool_name,
            parsed_confidence=parsed.confidence,
            parsed_approach=parsed.approach,
            parse_method=parsed.parse_method,
            all_matched_skills=parsed.all_matched_skills,
            expected_skills=expected_skills,
            skill_correct=skill_correct,
            error=error_msg,
        )

        self.results.append(result)
        self._completed_keys.add(f"{model_config.name}|{task.task_id}|{condition_name}|{temperature}")

        return result

    def run_batch(
        self,
        model_configs: list[ModelConfig],
        tasks: list,
        conditions: list,
        temperatures: list[float],
        skill_libraries: dict[str, list],
        valid_skill_names: list[str],
        resume_from: str | None = None,
        save_every: int = 10,
    ) -> list[RunResult]:
        """
        Run a full batch of evaluations.

        Args:
            model_configs: List of ModelConfig to evaluate.
            tasks: List of EvalTask objects.
            conditions: List of condition names (strings).
            temperatures: List of temperature values.
            skill_libraries: Dict mapping condition name -> list of Skill objects.
            valid_skill_names: List of all valid skill names for parsing.
            resume_from: Path to existing results file to resume from.
            save_every: Save results every N runs.

        Returns:
            List of RunResult objects.
        """
        # Resume from existing results if specified
        if resume_from:
            self._load_existing_results(resume_from)

        # Calculate total runs
        total_runs = len(model_configs) * len(tasks) * len(conditions) * len(temperatures)
        completed = len(self.results)
        remaining = total_runs - completed

        logger.info(f"Batch run: {len(model_configs)} models × {len(tasks)} tasks × "
                     f"{len(conditions)} conditions × {len(temperatures)} temps = {total_runs} total runs")
        logger.info(f"Already completed: {completed}, remaining: {remaining}")

        run_count = 0
        results_file = os.path.join(self.output_dir, "multi_model_results_latest.json")

        for model_config in model_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running model: {model_config.name} ({model_config.provider})")
            logger.info(f"{'='*60}")

            for temperature in temperatures:
                # Create config copy with updated temperature
                config_dict = asdict(model_config)
                config_dict['temperature'] = temperature
                temp_config = ModelConfig(**config_dict)

                for condition_name in conditions:
                    base_skill_lib = skill_libraries.get(condition_name, [])

                    for task in tasks:
                        # Oracle fix: show only the gold skill(s) for this task, not all 40 clean skills
                        if condition_name == "oracle":
                            gold_ids = set(task.expected_skill_ids)
                            skill_lib = [s for s in base_skill_lib if s.name in gold_ids or s.skill_id in gold_ids]
                        else:
                            skill_lib = base_skill_lib
                        try:
                            result = self.run_single(
                                model_config=temp_config,
                                task=task,
                                condition_name=condition_name,
                                skill_library=skill_lib,
                                temperature=temperature,
                                valid_skill_names=valid_skill_names,
                            )
                            run_count += 1

                            # Save incrementally
                            if run_count % save_every == 0:
                                self._save_results(results_file)
                                logger.info(f"  Progress: {completed + run_count}/{total_runs} runs")

                        except Exception as e:
                            logger.error(f"Error in run: {e}")
                            # Create error result
                            run_id = self._make_run_id(
                                model_config.name, task.task_id, condition_name, temperature
                            )
                            result = RunResult(
                                run_id=run_id,
                                model=model_config.name,
                                model_id=model_config.model_id,
                                task_id=task.task_id,
                                condition=condition_name,
                                temperature=temperature,
                                latency_ms=0,
                                timestamp=datetime.now().isoformat(),
                                prompt_tokens=0,
                                completion_tokens=0,
                                total_tokens=0,
                                prompt_text="",
                                raw_response="",
                                parsed_tool="PARSE_ERROR",
                                parsed_confidence=0.0,
                                parsed_approach="",
                                parse_method=PARSE_METHOD_ERROR,
                                expected_skills=task.expected_skill_ids if hasattr(task, 'expected_skill_ids') else [],
                                skill_correct=False,
                                error=str(e),
                            )
                            self.results.append(result)
                            run_count += 1

        # Final save
        final_file = self._save_results(results_file)
        logger.info(f"\nBatch complete! {len(self.results)} results saved to {final_file}")

        return self.results


def _normalize_match(name1: str, name2: str) -> bool:
    """Check if two skill names match after normalization."""
    def norm(n):
        return re.sub(r'[^a-z0-9]', '', n.lower())
    return norm(name1) == norm(name2)


# Keep regex import available for _normalize_match
import re


def produce_summary(results: list[RunResult], output_dir: str) -> dict:
    """Produce summary statistics from results and save to CSV and JSON."""
    import csv

    if not results:
        logger.warning("No results to summarize")
        return {}

    # Group by model × condition × temperature
    groups = {}
    for r in results:
        key = f"{r.model}|{r.condition}|T={r.temperature}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    summary = {}
    for key, group in groups.items():
        total = len(group)
        parsed = sum(1 for r in group if r.parse_method != PARSE_METHOD_ERROR)
        correct = sum(1 for r in group if r.skill_correct)
        errors = sum(1 for r in group if r.error)
        avg_latency = sum(r.latency_ms for r in group) / total if total else 0
        avg_confidence = sum(r.parsed_confidence for r in group) / total if total else 0
        total_tokens = sum(r.total_tokens for r in group)

        summary[key] = {
            "total_runs": total,
            "parsed": parsed,
            "parse_rate": round(parsed / total, 4) if total else 0,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0,
            "errors": errors,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_confidence": round(avg_confidence, 4),
            "total_tokens": total_tokens,
        }

    # Save CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "condition", "temperature", "total_runs", "parsed", "parse_rate",
                         "correct", "accuracy", "errors", "avg_latency_ms", "avg_confidence", "total_tokens"])
        for key, stats in summary.items():
            parts = key.split("|")
            model = parts[0]
            condition = parts[1]
            temp = parts[2].replace("T=", "")
            writer.writerow([
                model, condition, temp,
                stats["total_runs"], stats["parsed"], stats["parse_rate"],
                stats["correct"], stats["accuracy"], stats["errors"],
                stats["avg_latency_ms"], stats["avg_confidence"], stats["total_tokens"],
            ])

    # Save JSON
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {csv_path} and {json_path}")
    return summary


if __name__ == "__main__":
    # Quick test with mock setup
    from model_config import print_model_summary
    print_model_summary()