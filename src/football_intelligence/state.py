from typing import TypedDict


class Article(TypedDict):
    source_id: str

    url: str
    normalized_url: str

    domain: str
    title: str
    publisher: str
    author: str | None
    published_at: str | None
    retrieved_at: str

    article_text: str
    search_snippet: str
    search_score: float

    # Source strategy
    source_tier: str
    source_priority: int
    source_category: str
    source_search: str

    # Relevance
    relevance_score: float
    title_overlap: float
    snippet_overlap: float
    football_score: float
    unrelated_penalty: float
    matched_query_terms: list[str]
    relevance_rejected: bool

    # Quality
    content_length: int
    quality_score: float

    # Deduplication
    content_hash: str


class StoryClusterState(TypedDict):
    cluster_id: str
    representative_article: Article
    articles: list[Article]
    semantic_score: float
    independent_source_count: int
    domains: list[str]
    created_at: str
    relevance_classification: str
    relevance_reason: str


class ClaimEvidenceState(TypedDict):
    source_id: str
    evidence_type: str
    supporting_text: str
    domain: str
    source_priority: int
    origin: str
    provenance_confidence: str
    origin_kind: str


class ExtractedClaimState(TypedDict):
    story_cluster_id: str
    claim_text: str
    claim_type: str
    confidence: float
    evidence_status: str
    evidence_sources: list[ClaimEvidenceState]
    provenance_groups: list[str]
    independent_origin_count: int
    contradictions: list[dict]
    source_provenance: list[dict]


class ResearchState(TypedDict):
    query: str
    mode: str
    run_id: str
    max_results: int
    started_at: str

    articles: list[Article]
    story_clusters: list[StoryClusterState]
    claims: list[ExtractedClaimState]

    error: str | None
    status: str
    saved_count: int
