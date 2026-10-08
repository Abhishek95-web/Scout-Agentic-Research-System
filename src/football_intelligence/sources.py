from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CONFIG_PATH = PROJECT_ROOT / "config" / "sources.yaml"


@dataclass(frozen=True)
class SourcePolicy:
    domain: str
    priority: int
    enabled: bool
    category: str


def normalize_domain(domain: str) -> str:
    return domain.lower().strip().removeprefix("www.")


def load_source_policies() -> tuple[SourcePolicy, ...]:
    """
    Load source policies from config/sources.yaml.
    """

    if not SOURCE_CONFIG_PATH.exists():
        return ()

    with SOURCE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file) or {}

    policies = []

    for item in config.get("sources", []):
        domain = item.get("domain")

        if not domain:
            continue

        priority = int(item.get("priority", 0))

        enabled = bool(item.get("enabled", True))

        category = str(item.get("category", "general"))

        policies.append(
            SourcePolicy(
                domain=normalize_domain(domain),
                priority=priority,
                enabled=enabled,
                category=category,
            )
        )

    return tuple(policies)


def get_source_policy(
    domain: str,
) -> SourcePolicy | None:
    """
    Find the configured policy for a domain.

    Subdomains are supported.

    Example:

        sport.bbc.com

    matches:

        bbc.com
    """

    normalized = normalize_domain(domain)

    for policy in load_source_policies():
        if not policy.enabled:
            continue

        if normalized == policy.domain or normalized.endswith(f".{policy.domain}"):
            return policy

    return None


def get_source_tier(domain: str) -> str:
    """
    Determine the source tier.

    Configured sources are priority sources.
    Everything else remains standard discovery.
    """

    policy = get_source_policy(domain)

    if policy:
        return "priority"

    return "standard"


def get_source_priority(domain: str) -> int:
    """
    Return configured search priority.

    Unknown sources receive priority 0.
    """

    policy = get_source_policy(domain)

    if policy:
        return policy.priority

    return 0


def get_source_category(
    domain: str,
) -> str:
    """
    Return the configured source category.
    """

    policy = get_source_policy(domain)

    if policy:
        return policy.category

    return "unknown"


def get_priority_domains() -> list[str]:
    """
    Return all enabled domains configured
    as priority sources.
    """

    policies = load_source_policies()

    return [policy.domain for policy in policies if policy.enabled]
