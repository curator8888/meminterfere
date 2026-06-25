"""
MemInterfere: Model Configuration for Multi-Model Evaluation

Defines model backends, API endpoints, and rate limits for the 4-tier
model evaluation harness.

Tier layout:
  - Small:    Llama-3.1-8b-instruct  (OpenRouter, free)
  - Medium:   Grok-3-mini            (xAI, $0.30/M input)
  - Capable:  GPT-4o-mini            (OpenRouter, $0.15/M input)
  - Strong:   Claude-3.5-Haiku       (Anthropic, $0.25/M input)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single LLM backend."""
    name: str                          # Human-friendly name, e.g. "grok-3-mini"
    provider: str                      # "xai", "openrouter", "anthropic"
    model_id: str                      # API model identifier
    tier: str                          # "small", "medium", "capable", "strong"
    max_tokens: int                    # Max output tokens per request
    temperature: float                 # Default temperature
    supports_json_mode: bool           # Whether model supports response_format json_object
    cost_per_million_input: float      # USD per 1M input tokens
    cost_per_million_output: float     # USD per 1M output tokens
    rate_limit_rpm: int                # Requests per minute
    api_base: str                      # Base URL for the API
    env_key: str                       # Environment variable name for the API key
    extra_headers: dict = field(default_factory=dict)  # Extra HTTP headers


# ── Pre-configured model catalogue ──────────────────────────────────────────

MODELS: dict[str, ModelConfig] = {
    "llama-3.1-8b-instruct": ModelConfig(
        name="llama-3.1-8b-instruct",
        provider="openrouter",
        model_id="meta-llama/llama-3.1-8b-instruct",
        tier="small",
        max_tokens=2048,
        temperature=0.0,
        supports_json_mode=True,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
        rate_limit_rpm=20,
        api_base="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        extra_headers={
            "HTTP-Referer": "https://meminterfere.research",
            "X-Title": "MemInterfere Evaluation",
        },
    ),

    "grok-3-mini": ModelConfig(
        name="grok-3-mini",
        provider="xai",
        model_id="grok-3-mini",
        tier="medium",
        max_tokens=4096,
        temperature=0.0,
        supports_json_mode=True,
        cost_per_million_input=0.30,
        cost_per_million_output=0.50,
        rate_limit_rpm=60,
        api_base="https://api.x.ai/v1",
        env_key="XAI_API_KEY",
    ),

    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider="openrouter",
        model_id="openai/gpt-4o-mini",
        tier="capable",
        max_tokens=4096,
        temperature=0.0,
        supports_json_mode=True,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
        rate_limit_rpm=40,
        api_base="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        extra_headers={
            "HTTP-Referer": "https://meminterfere.research",
            "X-Title": "MemInterfere Evaluation",
        },
    ),

    "claude-3.5-haiku": ModelConfig(
        name="claude-3.5-haiku",
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        tier="strong",
        max_tokens=4096,
        temperature=0.0,
        supports_json_mode=False,  # Anthropic doesn't have JSON mode in the same way
        cost_per_million_input=0.25,
        cost_per_million_output=1.25,
        rate_limit_rpm=50,
        api_base="https://api.anthropic.com/v1",
        env_key="ANTHROPIC_API_KEY",
    ),
}


# ── Convenience lookups ─────────────────────────────────────────────────────

def get_model(name: str) -> ModelConfig:
    """Get a model config by name. Raises KeyError if not found."""
    if name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name]


def get_models_by_tier(tier: str) -> list[ModelConfig]:
    """Get all models for a given tier."""
    return [m for m in MODELS.values() if m.tier == tier]


def get_all_models() -> list[ModelConfig]:
    """Get all configured models."""
    return list(MODELS.values())


def estimate_cost(model: ModelConfig, n_runs: int, avg_input_tokens: int = 400,
                   avg_output_tokens: int = 150) -> float:
    """Estimate total cost in USD for a given number of runs."""
    input_cost = n_runs * avg_input_tokens * model.cost_per_million_input / 1_000_000
    output_cost = n_runs * avg_output_tokens * model.cost_per_million_output / 1_000_000
    return round(input_cost + output_cost, 4)


def print_model_summary():
    """Print a summary of all configured models and estimated costs."""
    total_runs = 175 * 5 * 3  # tasks × conditions × temperatures
    print("Model Configuration Summary")
    print("=" * 80)
    print(f"{'Model':<25} {'Tier':<10} {'Provider':<12} {'JSON':<5} {'RPM':<5} {'Cost/Run':<10} {'Est.Total':<10}")
    print("-" * 80)
    for model in MODELS.values():
        per_run = estimate_cost(model, 1)
        est_total = estimate_cost(model, total_runs)
        print(f"{model.name:<25} {model.tier:<10} {model.provider:<12} "
              f"{'✓' if model.supports_json_mode else '✗':<5} "
              f"{model.rate_limit_rpm:<5} ${per_run:<9.4f} ${est_total:<9.2f}")
    print(f"\nTotal estimated runs per model: {total_runs}")
    all_cost = sum(estimate_cost(m, total_runs) for m in MODELS.values())
    print(f"Estimated total cost (all 4 models): ${all_cost:.2f}")


if __name__ == "__main__":
    print_model_summary()