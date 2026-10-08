import re
from dataclasses import dataclass

FOOTBALL_TERMS = {
    "football",
    "soccer",
    "fifa",
    "uefa",
    "premier",
    "league",
    "champions",
    "europa",
    "club",
    "national",
    "squad",
    "manager",
    "coach",
    "player",
    "goal",
    "goals",
    "match",
    "fixture",
    "transfer",
    "transfers",
    "signing",
    "signed",
    "contract",
    "injury",
    "injured",
    "captain",
    "striker",
    "midfielder",
    "defender",
    "goalkeeper",
}

UNRELATED_TERMS = {
    "nfl",
    "nba",
    "nhl",
    "mlb",
    "cricket",
    "formula 1",
    "formula one",
    "f1",
    "tennis",
    "golf",
    "baseball",
    "basketball",
    "american football",
    "ncaa",
    "college football",
    "fantasy football",
    "quarterback",
    "touchdown",
    "super bowl",
    "cfp",
    "gridiron",
    "college football playoff",
}

# These words describe the request or sport, but rarely identify its subject.
GENERIC_TERMS = {
    "news",
    "team",
    "football",
    "soccer",
    "match",
    "game",
    "latest",
    "update",
    "updates",
    "report",
    "reports",
    "player",
    "players",
    "transfer",
    "transfers",
    "rumour",
    "rumours",
    "rumor",
    "rumors",
    "today",
    "the",
    "and",
    "for",
    "of",
    "in",
    "to",
}

# Small alias set for common names already represented in clustering.py.
# Canonical keys are used only to expand query matches; no entity model is loaded.
ENTITY_ALIASES = {
    "ronaldo": ("ronaldo", "cristiano ronaldo"),
    "cristiano": ("cristiano ronaldo", "ronaldo"),
    "manchester": ("manchester", "manchester united"),
    "united": ("manchester united", "united"),
    "real": ("real madrid", "madrid"),
    "madrid": ("real madrid", "madrid"),
    "juventus": ("juventus",),
    "portugal": ("portugal",),
    "arsenal": ("arsenal",),
    "barcelona": ("barcelona",),
    "norway": ("norway",),
    "uefa": ("uefa",),
    "fifa": ("fifa",),
}

# Contextual names already present in the project's conservative entity vocabulary.
# They can support a Portugal query when the country name is omitted in a headline.
CONTEXTUAL_ANCHORS = {
    "portugal": {"jorge jesus", "cristiano ronaldo", "ronaldo"},
}


@dataclass(frozen=True)
class RelevanceResult:
    score: float
    title_overlap: float
    snippet_overlap: float
    football_score: float
    unrelated_penalty: float
    matched_terms: tuple[str, ...]
    rejected: bool


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric terms."""
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_query_terms(query: str) -> set[str]:
    """Return query tokens of at least three characters (legacy API)."""
    return {token for token in tokenize(query) if len(token) >= 3}


def calculate_overlap(query_terms: set[str], text: str) -> tuple[float, set[str]]:
    if not query_terms:
        return 0.0, set()
    text_terms = set(tokenize(text))
    matched = query_terms & text_terms
    return len(matched) / len(query_terms), matched


def _term_weight(term: str) -> float:
    """Downweight generic vocabulary; distinctive terms become query anchors."""
    if term in GENERIC_TERMS:
        return 0.35
    # Longer, less common-looking terms tend to identify people, places or clubs.
    return 1.8 if len(term) >= 4 else 1.0


def _weighted_overlap(query_terms: set[str], text: str) -> tuple[float, set[str]]:
    if not query_terms:
        return 0.0, set()
    text_terms = set(tokenize(text))
    matched = query_terms & text_terms
    total_weight = sum(_term_weight(term) for term in query_terms)
    score = sum(_term_weight(term) for term in matched) / total_weight
    return score, matched


def _anchor_coverage(query_terms: set[str], text: str) -> tuple[float, set[str]]:
    anchors = {term for term in query_terms if term not in GENERIC_TERMS}
    if not anchors:
        # Generic queries such as "football news" still use ordinary overlap.
        return _weighted_overlap(query_terms, text)[0], set()
    text_lower = f" {text.lower()} "
    tokens = set(tokenize(text))
    matched: set[str] = set()
    for anchor in anchors:
        aliases = ENTITY_ALIASES.get(anchor, (anchor,))
        aliases = (*aliases, *CONTEXTUAL_ANCHORS.get(anchor, set()))
        if any(alias in text_lower for alias in aliases) or anchor in tokens:
            matched.add(anchor)
    return len(matched) / len(anchors), matched


def calculate_football_score(title: str, snippet: str) -> float:
    tokens = set(tokenize(f"{title} {snippet}"))
    matches = tokens & FOOTBALL_TERMS
    return min(1.0, len(matches) / 4) if matches else 0.0


def calculate_unrelated_penalty(title: str, snippet: str) -> float:
    text = f"{title} {snippet}".lower()
    return (
        0.8
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in UNRELATED_TERMS)
        else 0.0
    )


def assess_relevance(query: str, title: str, snippet: str) -> RelevanceResult:
    """Score a result with weighted query anchors and lightweight football signals."""
    query_terms = normalize_query_terms(query)
    title_overlap, title_matches = _weighted_overlap(query_terms, title)
    snippet_overlap, snippet_matches = _weighted_overlap(query_terms, snippet)
    anchor_score, anchor_matches = _anchor_coverage(query_terms, f"{title} {snippet}")
    title_anchor_score, _ = _anchor_coverage(query_terms, title)
    football_score = calculate_football_score(title, snippet)
    unrelated_penalty = calculate_unrelated_penalty(title, snippet)
    matched_terms = title_matches | snippet_matches | anchor_matches

    has_anchors = any(term not in GENERIC_TERMS for term in query_terms)
    if has_anchors:
        # Anchor agreement leads; title evidence outweighs snippet evidence.
        score = (
            anchor_score * 0.45
            + title_overlap * 0.30
            + snippet_overlap * 0.15
            + football_score * 0.10
            - unrelated_penalty
        )
        # Football vocabulary alone is weak evidence for an entity-specific query.
        if anchor_score == 0:
            score *= 0.35
        elif anchor_score < 1.0:
            score *= 0.85
        if title_anchor_score == 0:
            # Search snippets can contain navigation and unrelated headlines.
            score *= 0.35
    else:
        # Keep the original balance for broad football-news queries.
        score = (
            title_overlap * 0.50
            + snippet_overlap * 0.30
            + football_score * 0.20
            - unrelated_penalty
        )
    score = max(0.0, min(1.0, score))
    rejected = (unrelated_penalty >= 0.8 and title_overlap < 0.5) or (
        has_anchors and title_anchor_score == 0 and score < 0.24
    )
    return RelevanceResult(
        score=round(score, 4),
        title_overlap=round(title_overlap, 4),
        snippet_overlap=round(snippet_overlap, 4),
        football_score=round(football_score, 4),
        unrelated_penalty=round(unrelated_penalty, 4),
        matched_terms=tuple(sorted(matched_terms)),
        rejected=rejected,
    )
