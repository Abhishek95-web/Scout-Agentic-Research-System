import hashlib
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

import httpx
import trafilatura
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

from football_intelligence.database import connect_database, initialize_database
from football_intelligence.sources import (
    get_source_category,
    get_source_priority,
    get_source_tier,
)
from football_intelligence.state import (
    Article,
    ClaimEvidenceState,
    ExtractedClaimState,
    ResearchState,
)

from .claims import (
    ClaimCandidate,
    extract_claims_from_cluster,
    verify_claim,
)
from .clustering import cluster_articles
from .provenance import compare_claims, group_source_origins
from .quality import assess_quality
from .query_intent import classify_story_relevance
from .relevance import assess_relevance

_ASSOCIATION_FOOTBALL_SIGNAL = re.compile(
    r"\b(association football|soccer|football|fifa|uefa|aiff|cbf|"
    r"premier league|champions league|world cup|nations league|"
    r"national teams?|national side|football association|strikers?|midfielders?|"
    r"goalkeepers?|starting xi|lineups?|line-ups?|transfer market|transfers?|football club|"
    r"india super league|i-league|international friendly)\b",
    re.IGNORECASE,
)


def focused_football_query(query: str) -> str:
    """Keep the requested entities while disambiguating football as soccer."""
    if _ASSOCIATION_FOOTBALL_SIGNAL.search(query):
        return query
    subject_tokens = set(re.findall(r"[a-z0-9]+", query.lower())) - {
        "news",
        "team",
        "teams",
        "today",
        "latest",
        "update",
        "updates",
        "report",
        "reports",
        "match",
        "game",
        "football",
        "soccer",
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
    if len(subject_tokens) == 2:
        return f"{query} vs football match soccer"
    return f"{query} association football soccer"


def has_association_football_context(text: str) -> bool:
    return _ASSOCIATION_FOOTBALL_SIGNAL.search(text) is not None


def focused_result_matches_query(title: str, query: str) -> bool:
    """Require a focused headline to carry at least half of its subject anchors."""
    generic = {
        "football",
        "soccer",
        "news",
        "team",
        "teams",
        "today",
        "latest",
        "update",
        "updates",
        "match",
        "report",
        "reports",
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
    anchors = set(re.findall(r"[a-z0-9]+", query.lower())) - generic
    headline = set(re.findall(r"[a-z0-9]+", title.lower()))
    return not anchors or len(anchors & headline) / len(anchors) >= 0.5


def extract_claims(
    state: ResearchState,
) -> dict[str, list[ExtractedClaimState]]:
    """Extract and deduplicate candidate claims within each story cluster."""
    extracted: list[ExtractedClaimState] = []
    for cluster in state["story_clusters"]:
        groups = extract_claims_from_cluster(cluster["articles"])
        for group in groups:
            evidence_sources: list[ClaimEvidenceState] = []
            seen_source_ids: set[str] = set()
            for candidate in group.candidates:
                if candidate.source_id in seen_source_ids:
                    continue
                seen_source_ids.add(candidate.source_id)
                evidence_sources.append(
                    {
                        "source_id": candidate.source_id,
                        "evidence_type": "supports",
                        "supporting_text": candidate.supporting_text,
                        "domain": candidate.domain,
                        "source_priority": candidate.source_priority,
                        "origin": "",
                        "provenance_confidence": "low",
                        "origin_kind": "unknown",
                    }
                )
            extracted.append(
                {
                    "story_cluster_id": cluster["cluster_id"],
                    "claim_text": group.claim_text,
                    "claim_type": group.claim_type,
                    "confidence": group.confidence,
                    "evidence_status": "unverified",
                    "evidence_sources": evidence_sources,
                    "provenance_groups": [],
                    "independent_origin_count": 0,
                    "contradictions": [],
                }
            )
    logger.info("Candidate claims extracted: %d", len(extracted))
    return {"claims": extracted}


def link_evidence(
    state: ResearchState,
) -> dict[str, list[ExtractedClaimState]]:
    """Attach source-level support and assign a conservative first status."""
    for claim in state.get("claims", []):
        candidates = [
            ClaimCandidate(
                claim_text=claim["claim_text"],
                source_id=evidence["source_id"],
                confidence=claim["confidence"],
                claim_type=claim["claim_type"],
                supporting_text=evidence["supporting_text"],
                domain=evidence["domain"],
                source_priority=evidence["source_priority"],
                origin=evidence.get("origin", ""),
                provenance_confidence=evidence.get("provenance_confidence", "low"),
            )
            for evidence in claim["evidence_sources"]
            if evidence["evidence_type"] == "supports"
        ]
        claim["evidence_status"] = verify_claim(
            candidates,
            confidence=claim["confidence"],
        )
    return {"claims": state.get("claims", [])}


def analyze_provenance(state: ResearchState) -> dict:
    """Attach article origin estimates and recalculate source-aware statuses."""
    articles = {article["source_id"]: article for article in state.get("articles", [])}
    for claim in state.get("claims", []):
        sources = [
            articles[item["source_id"]]
            for item in claim["evidence_sources"]
            if item["source_id"] in articles
        ]
        groups = group_source_origins(sources)
        claim["provenance_groups"] = sorted(groups.groups)
        claim["independent_origin_count"] = groups.independent_origin_count
        by_id = {
            item.source_id: item for group in groups.groups.values() for item in group
        }
        claim["source_provenance"] = [
            {
                "source_id": item.source_id,
                "origin": item.origin,
                "confidence": item.confidence,
                "reason": item.reason,
                "kind": item.kind,
                "domain": item.domain,
            }
            for group in groups.groups.values()
            for item in group
        ]
        for evidence in claim["evidence_sources"]:
            origin = by_id.get(evidence["source_id"])
            if origin:
                evidence["origin"] = origin.origin
                evidence["provenance_confidence"] = origin.confidence
                evidence["origin_kind"] = origin.kind
        candidates = [
            ClaimCandidate(
                claim_text=claim["claim_text"],
                source_id=item["source_id"],
                confidence=claim["confidence"],
                claim_type=claim["claim_type"],
                supporting_text=item["supporting_text"],
                domain=item["domain"],
                source_priority=item["source_priority"],
                origin=item.get("origin", ""),
                provenance_confidence=item.get("provenance_confidence", "low"),
                origin_kind=item.get("origin_kind", "unknown"),
            )
            for item in claim["evidence_sources"]
            if item["evidence_type"] == "supports"
        ]
        claim["evidence_status"] = verify_claim(
            candidates, confidence=claim["confidence"]
        )
    return {"claims": state.get("claims", [])}


def analyze_contradictions(state: ResearchState) -> dict:
    """Compare claims within each story and mark only clear polarity conflicts."""
    claims = state.get("claims", [])
    for claim in claims:
        claim["contradictions"] = []
    for index, first in enumerate(claims):
        for other_index in range(index + 1, len(claims)):
            second = claims[other_index]
            if first["story_cluster_id"] != second["story_cluster_id"]:
                continue
            result = compare_claims(first["claim_text"], second["claim_text"])
            if result.is_contradiction:
                first["contradictions"].append(
                    {
                        "claim_index": other_index,
                        "reason": result.reason,
                        "confidence": result.confidence,
                    }
                )
                second["contradictions"].append(
                    {
                        "claim_index": index,
                        "reason": result.reason,
                        "confidence": result.confidence,
                    }
                )
                first["evidence_status"] = second["evidence_status"] = "disputed"
    return {"claims": claims}


def make_content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


load_dotenv()


logger = logging.getLogger(__name__)


USER_AGENT = (
    "FootballIntelligenceResearch/1.0 (personal research project; article retrieval)"
)


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_url(url: str) -> str:
    """Normalize URLs for duplicate detection."""

    parts = urlsplit(url.strip())

    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()

    if not scheme or not hostname:
        return url.strip()

    port = parts.port

    if port and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        hostname = f"{hostname}:{port}"

    path = parts.path.rstrip("/") or "/"

    query_items = []

    for key, value in parse_qsl(
        parts.query,
        keep_blank_values=True,
    ):
        key_lower = key.lower()

        if key_lower.startswith("utm_"):
            continue

        if key_lower in TRACKING_PARAMETERS:
            continue

        query_items.append((key, value))

    query = urlencode(
        query_items,
        doseq=True,
    )

    return urlunsplit(
        (
            scheme,
            hostname,
            path,
            query,
            "",
        )
    )


def get_domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def make_source_id(url: str) -> str:
    normalized = normalize_url(url)

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_text(text: str | None) -> str:
    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def fetch_article_text(
    url: str,
    timeout_seconds: int = 12,
) -> str:
    """
    Fetch and extract article text.

    If retrieval fails, the caller can fall
    back to the search result snippet.
    """

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout_seconds,
            headers={
                "User-Agent": USER_AGENT,
            },
        ) as client:
            response = client.get(url)

            response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "html" not in content_type:
            return ""

        extracted = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        return clean_text(extracted)[:15000]

    except (
        httpx.HTTPError,
        ValueError,
    ) as exc:
        logger.info(
            "Could not retrieve article %s: %s",
            url,
            exc,
        )

        return ""


def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to the project .env file."
        )

    return TavilyClient(api_key=api_key)


def initialize_run(
    state: ResearchState,
) -> dict:
    """Create a database record for this research run."""

    initialize_database()

    with connect_database() as connection:
        connection.execute(
            """
            INSERT INTO research_runs (
                run_id,
                query,
                status,
                started_at,
                completed_at,
                error_message
            )
            VALUES (?, ?, ?, ?, NULL, NULL)
            """,
            (
                state["run_id"],
                state["query"],
                "running",
                state["started_at"],
            ),
        )

    return {
        "status": "running",
    }


def search_tavily(
    client: TavilyClient,
    query: str,
    max_results: int,
    include_domains: list[str] | None = None,
    search_depth: str = "basic",
) -> list[dict]:
    """
    Execute one Tavily search.

    This abstraction lets us add other search
    providers later without rewriting the
    research workflow.
    """

    kwargs = {
        "query": query,
        "topic": "news",
        "search_depth": search_depth,
        "time_range": "day",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": "text",
    }

    if include_domains:
        kwargs["include_domains"] = include_domains

    response = client.search(**kwargs)

    return response.get(
        "results",
        [],
    )


def build_article(
    result: dict,
    source_search: str,
    query: str,
) -> Article | None:
    """
    Convert a Tavily result into our internal
    Article representation.
    """

    url = (result.get("url") or "").strip()

    if not url:
        return None

    normalized_url = normalize_url(url)

    domain = get_domain(url)

    if not domain:
        return None

    snippet = clean_text(result.get("content", ""))

    title = clean_text(result.get("title")) or "Untitled"

    # -------------------------
    # RELEVANCE
    # -------------------------

    relevance = assess_relevance(
        query=query,
        title=title,
        snippet=snippet,
    )

    # -------------------------
    # ARTICLE CONTENT
    # -------------------------

    raw_content = result.get("raw_content")

    if not isinstance(
        raw_content,
        str,
    ):
        raw_content = ""

    article_text = clean_text(raw_content)

    # Tavily sometimes returns
    # little/no article content.
    if len(article_text) < 500:
        fetched_text = fetch_article_text(url)

        if len(fetched_text) > len(article_text):
            article_text = fetched_text

    # Search snippet fallback
    if not article_text:
        article_text = snippet

    # -------------------------
    # QUALITY
    # -------------------------

    quality = assess_quality(
        article_text=article_text,
        title=title,
    )

    if not quality.acceptable:
        return None

    # -------------------------
    # CONTENT HASH
    # -------------------------

    content_hash = make_content_hash(article_text)

    # -------------------------
    # ARTICLE
    # -------------------------

    return {
        "source_id": make_source_id(url),
        "url": url,
        "normalized_url": normalized_url,
        "domain": domain,
        "title": title,
        "publisher": domain,
        "author": None,
        "published_at": result.get("published_date"),
        "retrieved_at": utc_now().isoformat(),
        "article_text": (article_text[:15000]),
        "search_snippet": snippet,
        "search_score": float(
            result.get(
                "score",
                0.0,
            )
        ),
        # Source strategy
        "source_tier": (get_source_tier(domain)),
        "source_priority": (get_source_priority(domain)),
        "source_category": (get_source_category(domain)),
        "source_search": source_search,
        # Relevance
        "relevance_score": (relevance.score),
        "title_overlap": (relevance.title_overlap),
        "snippet_overlap": (relevance.snippet_overlap),
        "football_score": (relevance.football_score),
        "unrelated_penalty": (relevance.unrelated_penalty),
        "matched_query_terms": list(relevance.matched_terms),
        "relevance_rejected": relevance.rejected,
        # Quality
        "content_length": (quality.content_length),
        "quality_score": (quality.score),
        # Deduplication
        "content_hash": content_hash,
    }


def search_news(
    state: ResearchState,
) -> dict:
    """
    Search priority sources first and then
    perform broad discovery.

    Priority sources are NOT the only sources
    considered.
    """

    try:
        client = get_tavily_client()

        target_results = state["max_results"]

        mode = state.get("mode", "normal")
        if mode == "focused":
            searches = [(focused_football_query(state["query"]), "focused")]
            depth = "advanced"
        else:
            lanes = [
                state["query"]
                if state["query"].strip().lower()
                not in {
                    "football",
                    "football news",
                    "latest football news",
                    "today's football news",
                    "football news today",
                }
                else "association football soccer latest news",
                "Premier League soccer news",
                "UEFA Champions League European football news",
                "international association football national teams news",
                "football transfer manager news soccer",
                "Indian football soccer news",
            ]
            searches = list(
                dict.fromkeys((q, f"discovery-{i + 1}") for i, q in enumerate(lanes))
            )
            depth = "basic"

        # --------------------------------
        # SEARCH 1
        # Configured priority sources
        # --------------------------------

        per_search = max(3, min(6, target_results // max(1, len(searches))))
        combined_results = []
        for search_query, lane in searches:
            results = search_tavily(
                client=client,
                query=search_query,
                max_results=per_search,
                search_depth=depth,
            )
            combined_results.extend((item, lane) for item in results)

        articles: list[Article] = []

        seen_urls: set[str] = set()

        for (
            result,
            source_search,
        ) in combined_results:
            try:
                article = build_article(
                    result,
                    source_search,
                    state["query"],
                )
                if article is None:
                    continue
                if article["title"].strip().casefold() == "instagram" or article[
                    "domain"
                ].casefold().endswith("instagram.com"):
                    logger.info(
                        "Rejected social-platform search result: %s", article["title"]
                    )
                    continue
                contamination = re.compile(
                    r"\b(nfl|ncaa|college football|american football|fantasy football|quarterback|touchdown|super bowl|cfp|gridiron)\b",
                    re.IGNORECASE,
                )
                text = f"{article['title']} {article['search_snippet']}"
                football_context = (
                    f"{text} {article.get('article_text', '')[:4000]}"
                    if state.get("mode", "normal") == "focused"
                    else text
                )
                if not has_association_football_context(football_context):
                    logger.info(
                        "Rejected result without association-football context: %s",
                        article["title"],
                    )
                    continue
                if state.get(
                    "mode", "normal"
                ) == "focused" and not focused_result_matches_query(
                    article["title"], state["query"]
                ):
                    logger.info(
                        "Rejected focused result without query anchors in headline: %s",
                        article["title"],
                    )
                    continue
                if contamination.search(text) and not re.search(
                    r"\b(soccer|association football|premier league|uefa|fifa|football club)\b",
                    text,
                    re.IGNORECASE,
                ):
                    logger.info(
                        "Rejected non-association football result: %s", article["title"]
                    )
                    continue
                if article["relevance_rejected"]:
                    logger.info(
                        "Rejected low-anchor relevance result: %s",
                        article["title"],
                    )
                    continue
                # Reject obviously unrelated results.
                if (
                    article["unrelated_penalty"] >= 0.8
                    and article["relevance_score"] < 0.5
                ):
                    logger.info(
                        "Rejected unrelated result: %s",
                        article["title"],
                    )
                    continue
                if article["relevance_score"] < 0.10:
                    logger.info(
                        "rejected low level relevance result: %s",
                        article["title"],
                    )
                    continue

            except Exception:
                logger.exception("Could not process search result")

                continue

            if article is None:
                continue
            if article["unrelated_penalty"] >= 0.8 and article["relevance_score"] < 0.5:
                logger.info(
                    "Rejected unrelated result: %s",
                    article["title"],
                )
                continue

            normalized_url = article["normalized_url"]

            if normalized_url in seen_urls:
                continue

            article_relevance = classify_story_relevance(
                state["query"],
                article["title"],
                article["search_snippet"],
                article["article_text"],
                article["relevance_score"],
            )
            article["query_relevance_classification"] = article_relevance.classification

            seen_urls.add(normalized_url)

            articles.append(article)

        # --------------------------------
        # Ranking for COLLECTION ONLY
        # --------------------------------
        #
        # This is NOT our final story ranking.
        #
        # We simply want priority sources
        # near the front of the collected set.

        unique_articles = {}
        for article in articles:
            key = article["content_hash"]

            existing = unique_articles.get(key)
            if existing is None:
                unique_articles[key] = article
                continue

            # keep teh better
            existing_rank = (
                existing["relevance_score"],
                existing["source_priority"],
                existing["search_score"],
            )
            new_rank = (
                article["relevance_score"],
                article["source_priority"],
                article["search_score"],
            )
            if new_rank > existing_rank:
                unique_articles[key] = article

        articles = list(unique_articles.values())

        articles.sort(
            key=lambda article: (
                {"primary": 2, "related": 1, "low_relevance": 0}.get(
                    article.get("query_relevance_classification", "related"), 0
                ),
                article["relevance_score"],
                article["quality_score"],
                article["source_priority"],
                article["search_score"],
            ),
            reverse=True,
        )
        articles = articles[:target_results]

        return {
            "articles": articles,
            "error": None,
        }

    except Exception as exc:
        logger.exception("Football news search failed")

        return {
            "articles": [],
            "error": (f"{type(exc).__name__}: {exc}"),
        }


def save_results(
    state: ResearchState,
) -> dict:
    """
    Persist collected sources and story clusters,
    then mark the research run complete or failed.
    """

    saved_count = 0
    completed_at = utc_now().isoformat()
    story_ids_by_cluster: dict[str, str] = {}

    with connect_database() as connection:
        # ---------------------------------------------------------
        # 1. Save all collected sources
        # ---------------------------------------------------------

        for article in state["articles"]:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO sources (
                    source_id,
                    url,
                    domain,
                    title,
                    publisher,
                    author,
                    published_at,
                    retrieved_at,
                    source_type,
                    article_text,
                    content_hash,
                    source_tier,
                    source_priority,
                    source_category
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article["source_id"],
                    article["normalized_url"],
                    article["domain"],
                    article["title"],
                    article["publisher"],
                    article["author"],
                    article["published_at"],
                    article["retrieved_at"],
                    "news_outlet",
                    article["article_text"],
                    article["content_hash"],
                    article["source_tier"],
                    article["source_priority"],
                    article["source_category"],
                ),
            )

            saved_count += cursor.rowcount

        # ---------------------------------------------------------
        # 2. Save story clusters
        # ---------------------------------------------------------

        for cluster in state["story_clusters"]:
            representative = cluster["representative_article"]

            story_id = str(uuid.uuid4())
            story_ids_by_cluster[cluster["cluster_id"]] = story_id
            story_claim_statuses = {
                claim["evidence_status"]
                for claim in state.get("claims", [])
                if claim["story_cluster_id"] == cluster["cluster_id"]
            }
            story_status = next(
                (
                    status
                    for status in ("disputed", "confirmed", "developing")
                    if status in story_claim_statuses
                ),
                "unverified",
            )

            connection.execute(
                """
                INSERT INTO stories (
                    story_id,
                    run_id,
                    title,
                    summary,
                    topic,
                    evidence_status,
                    importance_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    story_id,
                    state["run_id"],
                    representative["title"],
                    representative["search_snippet"],
                    state["query"],
                    story_status,
                    representative["relevance_score"],
                    cluster["created_at"],
                ),
            )

            # -----------------------------------------------------
            # 3. Connect every article in the cluster to the story
            # -----------------------------------------------------

            for article in cluster["articles"]:
                source_id = article["source_id"]

                source_cursor = connection.execute(
                    """
                    SELECT source_id
                    FROM sources
                    WHERE source_id = ?
                    """,
                    (source_id,),
                )

                source_row = source_cursor.fetchone()

                if source_row is None:
                    continue

                connection.execute(
                    """
                    INSERT OR IGNORE INTO story_sources (
                        story_id,
                        source_id
                    )
                    VALUES (?, ?)
                    """,
                    (
                        story_id,
                        source_row["source_id"],
                    ),
                )

        # ---------------------------------------------------------
        # 3. Save claims and source evidence links
        # ---------------------------------------------------------

        claim_ids = [str(uuid.uuid4()) for _ in state.get("claims", [])]
        for claim_index, claim in enumerate(state.get("claims", [])):
            story_id = story_ids_by_cluster.get(claim["story_cluster_id"])
            if story_id is None:
                continue

            claim_id = claim_ids[claim_index]
            notes = json.dumps(
                {
                    "claim_type": claim["claim_type"],
                    "confidence": claim["confidence"],
                    "provenance_groups": claim.get("provenance_groups", []),
                    "independent_origin_count": claim.get(
                        "independent_origin_count", 0
                    ),
                    "source_provenance": claim.get("source_provenance", []),
                    "contradictions": [
                        {
                            "claim_id": claim_ids[item["claim_index"]],
                            "reason": item["reason"],
                            "confidence": item["confidence"],
                        }
                        for item in claim.get("contradictions", [])
                    ],
                },
                sort_keys=True,
            )
            connection.execute(
                """
                INSERT INTO claims (
                    claim_id,
                    story_id,
                    claim_text,
                    evidence_status,
                    notes
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    claim_id,
                    story_id,
                    claim["claim_text"],
                    claim["evidence_status"],
                    notes,
                ),
            )

            for evidence in claim["evidence_sources"]:
                source_cursor = connection.execute(
                    "SELECT source_id FROM sources WHERE source_id = ?",
                    (evidence["source_id"],),
                )
                if source_cursor.fetchone() is None:
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO claim_evidence (
                        claim_id,
                        source_id,
                        evidence_type
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        claim_id,
                        evidence["source_id"],
                        evidence["evidence_type"],
                    ),
                )

        # ---------------------------------------------------------
        # 4. Mark research run as completed or failed
        # ---------------------------------------------------------

        status = "failed" if state["error"] else "completed"

        connection.execute(
            """
            UPDATE research_runs
            SET
                status = ?,
                completed_at = ?,
                error_message = ?
            WHERE run_id = ?
            """,
            (
                status,
                completed_at,
                state["error"],
                state["run_id"],
            ),
        )

    return {
        "saved_count": saved_count,
        "status": status,
    }


def cluster_stories(
    state: ResearchState,
) -> ResearchState:
    """
    Group collected articles into semantic
    story clusters.
    """

    clusters = cluster_articles(state["articles"])

    state["story_clusters"] = []
    for cluster in clusters:
        representative = cluster.representative_article
        relevance = classify_story_relevance(
            state["query"],
            representative["title"],
            " ".join(article["search_snippet"] for article in cluster.articles),
            " ".join(article["article_text"][:5000] for article in cluster.articles),
            max(article["relevance_score"] for article in cluster.articles),
        )
        state["story_clusters"].append(
            {
                "cluster_id": cluster.cluster_id,
                "representative_article": representative,
                "articles": cluster.articles,
                "semantic_score": cluster.semantic_score,
                "independent_source_count": cluster.independent_source_count,
                "domains": cluster.domains,
                "created_at": cluster.created_at,
                "relevance_classification": relevance.classification,
                "relevance_reason": relevance.reason,
            }
        )

    logger.info(
        "Story clusters created: %d",
        len(state["story_clusters"]),
    )

    for cluster in state["story_clusters"]:
        logger.info(
            "Story %s: %s | articles=%d | sources=%d",
            cluster["cluster_id"],
            cluster["representative_article"]["title"],
            len(cluster["articles"]),
            cluster["independent_source_count"],
        )

    return state


def build_research_graph():

    graph = StateGraph(ResearchState)

    graph.add_node(
        "initialize_run",
        initialize_run,
    )

    graph.add_node(
        "search_news",
        search_news,
    )
    graph.add_node(
        "cluster_stories",
        cluster_stories,
    )
    graph.add_node(
        "extract_claims",
        extract_claims,
    )
    graph.add_node(
        "link_evidence",
        link_evidence,
    )
    graph.add_node("analyze_provenance", analyze_provenance)
    graph.add_node("analyze_contradictions", analyze_contradictions)
    graph.add_node(
        "save_results",
        save_results,
    )

    graph.add_edge(
        START,
        "initialize_run",
    )

    graph.add_edge(
        "initialize_run",
        "search_news",
    )

    graph.add_edge(
        "search_news",
        "cluster_stories",
    )

    graph.add_edge(
        "cluster_stories",
        "extract_claims",
    )

    graph.add_edge(
        "extract_claims",
        "link_evidence",
    )

    graph.add_edge(
        "link_evidence",
        "analyze_provenance",
    )
    graph.add_edge("analyze_provenance", "analyze_contradictions")
    graph.add_edge(
        "analyze_contradictions",
        "save_results",
    )

    graph.add_edge(
        "save_results",
        END,
    )

    return graph.compile()


def run_research(
    query: str = "football news today",
    max_results: int = 20,
    mode: str = "normal",
) -> ResearchState:

    mode = mode.lower().replace("/", "_")
    if mode in {"research", "focused_research"}:
        mode = "focused"
    if mode not in {"normal", "focused"}:
        raise ValueError("mode must be 'normal' or 'focused'")

    graph = build_research_graph()

    initial_state: ResearchState = {
        "query": query,
        "mode": mode,
        "run_id": str(uuid.uuid4()),
        "max_results": max_results,
        "started_at": (utc_now().isoformat()),
        "articles": [],
        "story_clusters": [],
        "claims": [],
        "error": None,
        "status": "pending",
        "saved_count": 0,
    }

    return graph.invoke(initial_state)
