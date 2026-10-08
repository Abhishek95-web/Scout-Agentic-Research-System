"""Deterministic, deliberately approximate source provenance and claim conflicts."""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceOrigin:
    source_id: str
    origin: str
    confidence: str
    reason: str
    kind: str
    domain: str


@dataclass(frozen=True)
class OriginGroups:
    groups: dict[str, tuple[SourceOrigin, ...]]
    publisher_count: int
    independent_origin_count: int
    origin_count: int


@dataclass(frozen=True)
class Contradiction:
    is_contradiction: bool
    confidence: str
    reason: str


_ALIASES = {
    "associated press": "associated_press",
    "ap news": "associated_press",
    "ap": "associated_press",
    "reuters news": "reuters",
    "reuters": "reuters",
    "afp": "afp",
    "agence france-presse": "afp",
    "agence france presse": "afp",
    "getty": "getty",
}
_OFFICIAL = {"uefa.com": "uefa", "premierleague.com": "premier_league"}
_KNOWN = {
    "reuters.com": "reuters",
    "apnews.com": "associated_press",
    "bbc.com": "bbc",
    "bbc.co.uk": "bbc",
    "skysports.com": "sky_sports",
    "espn.com": "espn",
    "theathletic.com": "the_athletic",
    "theguardian.com": "the_guardian",
    "transfermarkt.com": "transfermarkt",
}


def _host(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"^https?://", "", value).split("/", 1)[0].split(":", 1)[0]
    return value.removeprefix("www.")


def _alias(value: str) -> str | None:
    cleaned = re.sub(r"[^a-z0-9 .-]", "", value.lower()).strip(" .-")
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    for alias, origin in _ALIASES.items():
        if len(alias) > 2 and re.search(rf"\b{re.escape(alias)}\b", cleaned):
            return origin
    return None


def infer_source_origin(source: Mapping[str, object]) -> SourceOrigin:
    """Infer origin from explicit article attribution, publisher, then domain."""
    source_id = str(source.get("source_id") or "")
    domain = _host(str(source.get("domain") or ""))
    publisher = str(source.get("publisher") or "")
    title = str(source.get("title") or "")
    text = " ".join(
        str(source.get(k) or "") for k in ("article_text", "search_snippet")
    )
    evidence_text = text[:30000]
    explicit = _alias(evidence_text)
    if explicit is None and re.search(
        r"\b(?:according to|via|reported by|from)\s+ap\b",
        evidence_text,
        re.IGNORECASE,
    ):
        explicit = "associated_press"
    if explicit:
        return SourceOrigin(
            source_id,
            explicit,
            "high",
            f"explicit {explicit.replace('_', ' ')} attribution",
            "agency",
            domain,
        )
    label = _alias(publisher) or _alias(title)
    if label:
        return SourceOrigin(
            source_id,
            label,
            "high",
            "publisher identifies reporting origin",
            "agency",
            domain,
        )
    if domain in _OFFICIAL:
        origin = _OFFICIAL[domain]
        return SourceOrigin(
            source_id, origin, "high", "configured official source", "official", domain
        )
    for suffix, origin in _KNOWN.items():
        if domain == suffix or domain.endswith("." + suffix):
            kind = "official" if suffix in _OFFICIAL else "journalism"
            return SourceOrigin(
                source_id, origin, "high", "recognized publisher domain", kind, domain
            )
    fallback = (
        re.sub(r"[^a-z0-9]+", "_", publisher.lower()).strip("_")
        if publisher
        else domain.replace(".", "_")
    )
    return SourceOrigin(
        source_id,
        fallback or "unknown",
        "low",
        "no explicit origin detected; publisher/domain fallback",
        "unknown",
        domain,
    )


def group_source_origins(sources: Sequence[Mapping[str, object]]) -> OriginGroups:
    """Group source records by inferred origin; counts are not certainty scores."""
    grouped: dict[str, list[SourceOrigin]] = defaultdict(list)
    domains: set[str] = set()
    for source in sources:
        origin = infer_source_origin(source)
        grouped[origin.origin].append(origin)
        if origin.domain:
            domains.add(origin.domain)
    independent_count = sum(
        1 for items in grouped.values() if items and items[0].kind != "official"
    )
    return OriginGroups(
        {key: tuple(items) for key, items in grouped.items()},
        len(domains),
        independent_count,
        len(grouped),
    )


def normalize_claim(text: str) -> str:
    """Normalize case and punctuation while retaining negation and numbers."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def compare_claims(first: str, second: str) -> Contradiction:
    """Detect only a small set of clear opposing football propositions."""
    a, b = normalize_claim(first), normalize_claim(second)
    if a == b or not a or not b:
        return Contradiction(False, "low", "same or empty normalized claim")
    tokens_a, tokens_b = set(a.split()), set(b.split())
    entities = {
        "ronaldo",
        "portugal",
        "norway",
        "england",
        "france",
        "spain",
        "arsenal",
        "chelsea",
        "liverpool",
        "manchester",
        "haaland",
    }
    if not (tokens_a & tokens_b & entities):
        return Contradiction(False, "low", "no shared recognized subject")
    patterns = [
        (
            r"\bwill return\b",
            r"\bwill not return\b|\bwon t return\b",
            "opposing future-return propositions",
        ),
        (
            r"\bavailable\b",
            r"\bunavailable\b|\bnot available\b",
            "available versus unavailable",
        ),
        (
            r"\bselected\b",
            r"\bnot selected\b|\bunselected\b",
            "selected versus not selected",
        ),
        (
            r"\binjured\b",
            r"\bnot injured\b|\buninjured\b",
            "injured versus not injured",
        ),
        (
            r"\bwon\b|\bbeat\b|\bdefeated\b",
            r"\blost\b|\bloss to\b",
            "opposing match outcomes",
        ),
        (
            r"\bsigned\b",
            r"\bdid not sign\b|\bhas not signed\b",
            "signed versus not signed",
        ),
        (
            r"\bappointed\b",
            r"\bnot appointed\b|\bdid not appoint\b",
            "appointed versus not appointed",
        ),
    ]
    for positive, negative, reason in patterns:
        pos_a, neg_a = bool(re.search(positive, a)), bool(re.search(negative, a))
        pos_b, neg_b = bool(re.search(positive, b)), bool(re.search(negative, b))
        if (pos_a and neg_b) or (neg_a and pos_b):
            return Contradiction(True, "high", reason)
    return Contradiction(False, "low", "different but compatible claims")
