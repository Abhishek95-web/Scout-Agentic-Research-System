"""Deterministic claim extraction and conservative first-pass status logic."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

CLAIM_TYPES = {
    "event",
    "statement",
    "result",
    "transfer",
    "injury",
    "selection",
    "future",
    "other",
}

_ACTION_TERMS = {
    "beat",
    "beats",
    "beaten",
    "defeat",
    "defeated",
    "win",
    "wins",
    "won",
    "lose",
    "loses",
    "lost",
    "draw",
    "drew",
    "advance",
    "advanced",
    "reach",
    "reached",
    "qualify",
    "qualified",
    "score",
    "scored",
    "said",
    "says",
    "told",
    "announce",
    "announced",
    "confirm",
    "confirmed",
    "select",
    "selected",
    "include",
    "included",
    "omit",
    "omitted",
    "sign",
    "signed",
    "join",
    "joins",
    "joined",
    "leave",
    "left",
    "return",
    "returns",
    "returned",
    "injured",
    "injury",
    "rule",
    "ruled",
    "prepare",
    "prepares",
    "prepared",
    "face",
    "faces",
    "facing",
    "will",
    "could",
    "may",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "welcomes",
    "discusses",
    "expects",
    "plans",
    "set",
    "make",
    "makes",
    "exits",
    "exit",
}

_BOILERPLATE = re.compile(
    r"\b(read more|subscribe|watch live|watch now|click here|sign up|play super 6|"
    r"cookie|advertisement|advertising|share this|follow us|privacy policy|"
    r"terms and conditions|all rights reserved|skip to content|menu|newsletter|"
    r"terms of use|contact us|accessibility|download the app|unlock the latest)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_NON_CLAIM_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "as",
    "after",
    "before",
    "over",
    "under",
    "into",
    "their",
    "his",
    "her",
    "its",
    "they",
    "them",
    "he",
    "she",
    "we",
    "you",
    "i",
    "it",
    "this",
    "that",
    "who",
    "which",
    "what",
    "when",
    "where",
    "why",
    "how",
    "said",
    "says",
    "told",
    "beat",
    "beats",
    "beaten",
    "defeat",
    "defeated",
    "win",
    "wins",
    "won",
    "lose",
    "loses",
    "lost",
    "draw",
    "drew",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "will",
    "could",
    "may",
    "should",
    "would",
    "can",
    "must",
    "not",
    "no",
    "new",
    "latest",
    "news",
    "football",
    "team",
    "squad",
    "club",
    "match",
    "game",
    "today",
    "yesterday",
}
_KNOWN_ENTITIES = {
    "portugal",
    "norway",
    "england",
    "scotland",
    "ireland",
    "israel",
    "wales",
    "denmark",
    "france",
    "spain",
    "germany",
    "italy",
    "croatia",
    "netherlands",
    "uefa",
    "fifa",
    "arsenal",
    "chelsea",
    "liverpool",
    "manchester united",
    "manchester city",
    "real madrid",
    "barcelona",
    "juventus",
    "al nassr",
    "premier league",
    "champions league",
    "nations league",
    "cristiano ronaldo",
    "ronaldo",
    "jorge jesus",
    "erling haaland",
    "haaland",
    "goncalo ramos",
    "joao cancelo",
}


@dataclass(frozen=True)
class ClaimCandidate:
    """One source's candidate claim and its originating article evidence."""

    claim_text: str
    source_id: str
    confidence: float
    claim_type: str
    supporting_text: str
    domain: str
    source_priority: int
    origin: str = ""
    provenance_confidence: str = "low"
    origin_kind: str = "unknown"


@dataclass
class ClaimGroup:
    """Similar claim candidates grouped with their source-level evidence."""

    claim_text: str
    claim_type: str
    confidence: float
    candidates: list[ClaimCandidate] = field(default_factory=list)

    @property
    def source_ids(self) -> list[str]:
        """Return the unique sources that supplied this claim."""
        return list(dict.fromkeys(item.source_id for item in self.candidates))


def _claim_type(text: str) -> str:
    """Assign one of the small deterministic claim categories."""
    value = text.lower()
    if re.search(r"\b(said|says|told|according to|commented|stated)\b", value):
        return "statement"
    if re.search(r"\b(injur(?:y|ed)|ruled out|limped|fitness)\b", value):
        return "injury"
    if re.search(r"\b(transfer|signed|signing|joined|joins|loan|deal)\b", value):
        return "transfer"
    if re.search(
        r"\b(line[- ]?up|starting xi|squad|selected|selection|omitted|called up)\b",
        value,
    ):
        return "selection"
    if re.search(
        r"\b\d+\s*[-–:]\s*\d+\b|\b(defeated|beat|won|lost|drew|draw)\b", value
    ):
        return "result"
    if re.search(r"\b(could|may|might|will|future|return|expects|plans to)\b", value):
        return "future"
    if re.search(
        r"\b(advanced|reached|qualified|clash|face|prepare|prepared)\b", value
    ):
        return "event"
    return "other"


def _has_subject(text: str, tokens: list[str]) -> bool:
    generic_openers = {
        "a",
        "an",
        "the",
        "read",
        "watch",
        "subscribe",
        "latest",
        "breaking",
        "live",
        "after",
        "before",
        "meanwhile",
        "published",
        "play",
        "home",
        "here",
        "image",
        "article",
        "advertisement",
        "with",
        "when",
        "he",
        "she",
        "they",
        "it",
        "this",
        "that",
        "what",
        "why",
        "who",
        "how",
        "where",
        "which",
    }
    if tokens[0] in generic_openers:
        return False
    first_words = " ".join(tokens[:4])
    if any(entity in first_words for entity in _KNOWN_ENTITIES):
        return True
    # Allow an unfamiliar club, player or coach as the grammatical subject.
    return bool(re.match(r"^[A-Z][\w’'-]+\s+\w+", text))


def _shares_title_entity(text: str, title: str) -> bool:
    """Avoid extracting unrelated body/sidebar sentences from a relevant page."""
    title_lower = title.lower()
    title_entities = {entity for entity in _KNOWN_ENTITIES if entity in title_lower}
    if not title_entities:
        return True
    sentence_lower = text.lower()
    return any(entity in sentence_lower for entity in title_entities)


def _is_candidate(text: str, allow_headline: bool = False) -> bool:
    stripped = text.strip(" \t\r\n-–—•")
    if not stripped or len(stripped) > 360 or _BOILERPLATE.search(stripped):
        return False
    tokens = _TOKEN_RE.findall(stripped.lower())
    if len(tokens) < 5 or len(tokens) > 35:
        return False
    if tokens[0] in {"what", "why", "who", "when", "where", "how", "which"}:
        return False
    if not any(token in _ACTION_TERMS for token in tokens):
        return False
    if allow_headline and not stripped.endswith((".", "?", "!")):
        first_action = next(
            index for index, token in enumerate(tokens) if token in _ACTION_TERMS
        )
        if first_action > 2:
            return False
    if not _has_subject(stripped, tokens):
        return False
    meaningful = [token for token in tokens if token not in _NON_CLAIM_WORDS]
    return len(set(meaningful)) >= 2


def _clean_title(text: str) -> str:
    """Remove common publisher labels from headline-based claim text."""
    text = text.split("|", maxsplit=1)[0].strip()
    return re.sub(
        r"\s+-\s+(Sky Sports|ESPN|BBC(?: Sport)?|Al Jazeera|Yahoo Sports|"
        r"UEFA(?:\.com)?|Goal\.com)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _confidence(text: str, source_kind: str) -> float:
    value = 0.70 if source_kind == "title" else 0.74
    if text.rstrip().endswith((".", "?", "!")):
        value += 0.08
    if len(_TOKEN_RE.findall(text)) >= 9:
        value += 0.05
    if re.search(r"\b\d+\s*[-–:]\s*\d+\b", text):
        value += 0.08
    return min(value, 0.95)


def extract_claims_from_article(article: dict) -> list[ClaimCandidate]:
    """Conservatively extract useful declarative candidates from one article."""
    source_id = str(article.get("source_id") or "")
    if not source_id:
        return []
    title = str(article.get("title") or "")
    article_text = str(article.get("article_text") or "")
    snippet = str(article.get("search_snippet") or "")
    fields = [("title", title)]
    if article_text.strip():
        fields.append(("article", article_text))
    if not article_text.strip() or article_text.strip() == snippet.strip():
        fields.append(("snippet", snippet))
    output: list[ClaimCandidate] = []
    seen: set[str] = set()
    for kind, content in fields:
        for sentence in _SENTENCE_RE.split(content[:15000]):
            text = re.sub(r"\s+", " ", sentence).strip(" \t\r\n-–—•")
            if kind == "title":
                text = _clean_title(text)
            normalized = normalize_claim_text(text)
            if (
                normalized in seen
                or (
                    kind != "title"
                    and normalized == normalize_claim_text(_clean_title(title))
                    and not title.rstrip().endswith((".", "?", "!"))
                )
                or not _is_candidate(text, allow_headline=kind == "title")
                or (kind != "title" and not _shares_title_entity(text, title))
            ):
                continue
            seen.add(normalized)
            output.append(
                ClaimCandidate(
                    claim_text=text,
                    source_id=source_id,
                    confidence=_confidence(text, kind),
                    claim_type=_claim_type(text),
                    supporting_text=text,
                    domain=str(article.get("domain") or ""),
                    source_priority=int(article.get("source_priority") or 0),
                )
            )
            if len(output) >= 2:
                return output
    return output


def normalize_claim_text(text: str) -> str:
    """Normalize punctuation and common result synonyms for deduplication."""
    tokens = _TOKEN_RE.findall(text.lower())
    normalized = [
        "won" if token in {"beat", "beats", "defeated", "defeat"} else token
        for token in tokens
    ]
    return " ".join(normalized)


def _claim_similarity(first: ClaimCandidate, second: ClaimCandidate) -> float:
    first_tokens = (
        set(normalize_claim_text(first.claim_text).split()) - _NON_CLAIM_WORDS
    )
    second_tokens = (
        set(normalize_claim_text(second.claim_text).split()) - _NON_CLAIM_WORDS
    )
    union = first_tokens | second_tokens
    overlap = len(first_tokens & second_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(
        None,
        normalize_claim_text(first.claim_text),
        normalize_claim_text(second.claim_text),
    ).ratio()
    return max(overlap, sequence)


def deduplicate_claims(candidates: Sequence[ClaimCandidate]) -> list[ClaimGroup]:
    """Group semantically similar candidates using normalized lexical similarity."""
    groups: list[ClaimGroup] = []
    for candidate in candidates:
        matching_group = None
        for group in groups:
            representative = group.candidates[0]
            if (
                candidate.claim_type == group.claim_type
                and _claim_similarity(candidate, representative) >= 0.62
            ):
                matching_group = group
                break
        if matching_group is None:
            groups.append(
                ClaimGroup(
                    claim_text=candidate.claim_text,
                    claim_type=candidate.claim_type,
                    confidence=candidate.confidence,
                    candidates=[candidate],
                )
            )
            continue
        if candidate.source_id not in matching_group.source_ids:
            matching_group.candidates.append(candidate)
        if candidate.confidence > matching_group.confidence:
            matching_group.claim_text = candidate.claim_text
            matching_group.confidence = candidate.confidence
    return groups


def extract_claims_from_cluster(articles: Sequence[dict]) -> list[ClaimGroup]:
    """Extract and deduplicate claims across the articles in one story cluster."""
    if not articles:
        return []
    lead_title = str(articles[0].get("title") or "")
    candidates = [
        candidate
        for article in articles
        for candidate in extract_claims_from_article(article)
        if _shares_title_entity(candidate.claim_text, lead_title)
    ]
    return deduplicate_claims(candidates)


def verify_claim(
    supporting_sources: Sequence[ClaimCandidate],
    contradicting_source_count: int = 0,
    confidence: float = 0.0,
) -> str:
    """Assign a conservative status using distinct reporting origins."""
    support_count = len({item.source_id for item in supporting_sources})
    highest_priority = max(
        (item.source_priority for item in supporting_sources),
        default=0,
    )
    if support_count and contradicting_source_count:
        return "disputed"
    if not support_count:
        return "unverified"
    if confidence < 0.55 and highest_priority < 80:
        return "unverified"
    if support_count == 1:
        return "developing"
    origins = {item.origin for item in supporting_sources if item.origin}
    if len(origins) > 1 and highest_priority >= 90 and confidence >= 0.8:
        origin_confidences = {item.provenance_confidence for item in supporting_sources}
        # Official/primary corroboration can support confirmation, but counts alone cannot.
        has_official_source = any(
            item.origin_kind == "official" for item in supporting_sources
        )
        if (
            "high" in origin_confidences
            and has_official_source
            and len({item.source_id for item in supporting_sources}) >= 2
        ):
            return "confirmed"
    return "developing"
