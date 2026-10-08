"""Small query-anchor classifier for keeping reports focused on user intent."""

import re
from dataclasses import dataclass

_GENERIC = {
    "football",
    "soccer",
    "news",
    "today",
    "team",
    "teams",
    "latest",
    "update",
    "updates",
    "report",
    "reports",
    "story",
    "stories",
    "match",
    "game",
    "the",
    "and",
    "for",
    "with",
    "from",
    "about",
    "to",
    "in",
    "on",
    "vs",
    "versus",
}
_RELATED_CUES = {
    "follow",
    "following",
    "after",
    "reaction",
    "response",
    "consequence",
    "tickets",
    "ticket",
    "attendance",
    "turnout",
    "impact",
    "because",
    "amid",
}


@dataclass(frozen=True)
class StoryRelevance:
    classification: str
    reason: str
    matched_anchors: tuple[str, ...]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def classify_story_relevance(
    query: str,
    title: str,
    snippet: str = "",
    article_text: str = "",
    relevance_score: float = 0.0,
) -> StoryRelevance:
    """Classify a story as primary, related, or low relevance.

    Exact query anchors in the headline are strongest. Related text can support
    a follow-up classification, but cannot turn an unrelated headline primary.
    """
    anchors = _tokens(query) - _GENERIC
    title_tokens = _tokens(title)
    evidence_tokens = title_tokens | _tokens(snippet) | _tokens(article_text[:10000])
    matched = anchors & evidence_tokens
    title_matches = anchors & title_tokens
    if not anchors:
        label = (
            "primary"
            if relevance_score >= 0.35
            else "related"
            if relevance_score >= 0.18
            else "low_relevance"
        )
        return StoryRelevance(label, "broad query relevance", ())

    title_coverage = len(title_matches) / len(anchors)
    full_coverage = len(matched) / len(anchors)
    title_is_follow_up = bool(title_tokens & _RELATED_CUES)
    if title_coverage >= 0.5 and relevance_score >= 0.2 and not title_is_follow_up:
        return StoryRelevance(
            "primary",
            "headline directly matches the query anchors",
            tuple(sorted(title_matches)),
        )

    related_context = bool(title_tokens & _RELATED_CUES) or bool(
        evidence_tokens & _RELATED_CUES
    )
    if full_coverage >= 0.65 and title_coverage >= 0.35 and related_context:
        return StoryRelevance(
            "related",
            "headline is a follow-up connected to the query topic",
            tuple(sorted(matched)),
        )
    if full_coverage == 1 and title_coverage >= 0.5 and relevance_score >= 0.32:
        return StoryRelevance(
            "primary",
            "headline and article directly concern the requested topic",
            tuple(sorted(matched)),
        )
    if full_coverage >= 0.5 and relevance_score >= 0.22:
        return StoryRelevance(
            "related",
            "shares query anchors but is not the central event",
            tuple(sorted(matched)),
        )
    return StoryRelevance(
        "low_relevance",
        "insufficient direct query-anchor overlap",
        tuple(sorted(matched)),
    )
