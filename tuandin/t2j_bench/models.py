"""Model roster and cost-tier definitions for T2J-Bench pipeline.

Two tiers share the same schema so results from colleague (cheap) and
principal (premium) runs merge into one dataset.

Prices are USD per 1M tokens (input, output). Snapshot: 2026-08-15.
Re-check openrouter.ai/models before large runs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    slug: str
    family: str
    price_in: float
    price_out: float

    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1e6


WEAK = {
    "granite-8b":     Model("ibm-granite/granite-4.1-8b",       "IBM",       0.05, 0.10),
    "llama-8b":       Model("meta-llama/llama-3.1-8b-instruct", "Meta",      0.05, 0.08),
    "gemma-12b":      Model("google/gemma-3-12b-it",            "Google",    0.05, 0.15),
    "phi-4":          Model("microsoft/phi-4",                  "Microsoft", 0.07, 0.14),
    "gpt-oss-20b":    Model("openai/gpt-oss-20b",               "OpenAI",    0.03, 0.13),
    "qwen25-7b":      Model("qwen/qwen-2.5-7b-instruct",        "Alibaba",   0.10, 0.20),
    "qwen3-8b":       Model("qwen/qwen3-8b",                    "Alibaba",   0.12, 0.46),
    "qwen3-14b":      Model("qwen/qwen3-14b",                   "Alibaba",   0.12, 0.24),
    "ministral-8b":   Model("mistralai/ministral-8b-2512",      "Mistral",   0.15, 0.15),
    "codestral":      Model("mistralai/codestral-2508",         "Mistral",   0.30, 0.90),
    "qwen3-coder-flash": Model("qwen/qwen3-coder-flash",        "Alibaba",   0.20, 0.98),
}

STRONG = {
    "glm-5.2":        Model("z-ai/glm-5.2",                     "Zhipu",     0.46, 1.45),
    "kimi-k27-code":  Model("moonshotai/kimi-k2.7-code",        "Moonshot",  0.71, 3.50),
    "deepseek-v4":    Model("deepseek/deepseek-v4-pro",         "DeepSeek",  1.17, 2.34),
    "gemini-25-pro":  Model("google/gemini-2.5-pro",            "Google",    1.25, 10.00),
    "grok-4.6":       Model("x-ai/grok-4.6",                    "xAI",       2.00, 6.00),
    "o3":             Model("openai/o3",                        "OpenAI",    2.00, 8.00),
    "gemini-31-pro":  Model("google/gemini-3.1-pro-preview",    "Google",    2.00, 12.00),
    "gpt-5.4":        Model("openai/gpt-5.4",                   "OpenAI",    2.50, 15.00),
    "sonnet-4.6":     Model("anthropic/claude-sonnet-4.6",      "Anthropic", 3.00, 15.00),
    "kimi-k3":        Model("moonshotai/kimi-k3",               "Moonshot",  3.00, 15.00),
    "opus-4.8":       Model("anthropic/claude-opus-4.8",        "Anthropic", 5.00, 25.00),
    "gpt-5.6-sol-pro": Model("openai/gpt-5.6-sol-pro",          "OpenAI",    5.00, 30.00),
}

JUDGE = {
    "gpt-5-mini":     Model("openai/gpt-5-mini",                "OpenAI",    0.25, 2.00),
    "gemini-25-flash": Model("google/gemini-2.5-flash",         "Google",    0.30, 2.50),
    "gpt-54-mini":    Model("openai/gpt-5.4-mini",              "OpenAI",    0.75, 4.50),
    "gemini-35-flash": Model("google/gemini-3.5-flash",         "Google",    1.50, 9.00),
}


@dataclass(frozen=True)
class Tier:
    name: str
    weak_translators: tuple[str, ...]   # keys into WEAK
    junior_fixer: str                    # key into STRONG
    senior_fixer: str                    # key into STRONG
    rc_judge: str                        # key into JUDGE

    def resolve(self) -> dict:
        return {
            "weak": [WEAK[k] for k in self.weak_translators],
            "junior": STRONG[self.junior_fixer],
            "senior": STRONG[self.senior_fixer],
            "judge": JUDGE[self.rc_judge],
        }


CHEAP = Tier(
    name="cheap",
    weak_translators=("granite-8b", "llama-8b", "gemma-12b", "phi-4", "gpt-oss-20b"),
    junior_fixer="glm-5.2",
    senior_fixer="deepseek-v4",
    rc_judge="gemini-25-flash",
)

PREMIUM = Tier(
    name="premium",
    weak_translators=(
        "granite-8b", "llama-8b", "gemma-12b", "phi-4", "gpt-oss-20b",
        "qwen25-7b", "codestral", "qwen3-coder-flash",
    ),
    junior_fixer="sonnet-4.6",
    senior_fixer="opus-4.8",
    rc_judge="gemini-35-flash",   # Google, independent from Anthropic fixers
)

TIERS = {"cheap": CHEAP, "premium": PREMIUM}


def check_independence(tier: Tier) -> list[str]:
    """Return list of independence-constraint violations (empty = OK)."""
    r = tier.resolve()
    warnings = []
    if r["junior"].family == r["senior"].family:
        warnings.append(
            f"junior and senior fixer share family {r['junior'].family!r}; "
            "reviewers may flag as circular"
        )
    if r["judge"].family in {r["junior"].family, r["senior"].family}:
        warnings.append(
            f"RC judge family {r['judge'].family!r} overlaps with fixer family; "
            "use a third-family judge"
        )
    return warnings
