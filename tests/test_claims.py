from typing import cast

from football_intelligence.claims import (
    ClaimCandidate,
    deduplicate_claims,
    extract_claims_from_article,
    extract_claims_from_cluster,
    verify_claim,
)
from football_intelligence.database import connect_database, initialize_database
from football_intelligence.research import extract_claims, link_evidence, save_results
from football_intelligence.state import Article, ResearchState


def _article(
    source_id: str,
    title: str,
    article_text: str = "",
    snippet: str = "",
    domain: str = "example.com",
) -> Article:
    return cast(
        Article,
        {
            "source_id": source_id,
            "url": f"https://{domain}/{source_id}",
            "normalized_url": f"https://{domain}/{source_id}",
            "domain": domain,
            "title": title,
            "publisher": domain,
            "author": None,
            "published_at": "2026-10-05T10:00:00Z",
            "retrieved_at": "2026-10-05T11:00:00Z",
            "article_text": article_text,
            "search_snippet": snippet,
            "search_score": 0.8,
            "source_tier": "standard",
            "source_priority": 0,
            "source_category": "unknown",
            "source_search": "broad",
            "relevance_score": 0.8,
            "title_overlap": 0.8,
            "snippet_overlap": 0.8,
            "football_score": 0.8,
            "unrelated_penalty": 0.0,
            "matched_query_terms": ["portugal"],
            "relevance_rejected": False,
            "content_length": len(article_text),
            "quality_score": 0.8,
            "content_hash": source_id,
        },
    )


def _candidate(source_id: str, domain: str = "example.com") -> ClaimCandidate:
    return ClaimCandidate(
        claim_text="Portugal defeated Norway 2-1.",
        source_id=source_id,
        confidence=0.9,
        claim_type="result",
        supporting_text="Portugal defeated Norway 2-1.",
        domain=domain,
        source_priority=80,
    )


def test_extract_event_result_claim():
    claims = extract_claims_from_article(
        _article("source-a", "Portugal defeated Norway 2-1.")
    )

    assert claims
    assert claims[0].claim_type == "result"
    assert "Norway" in claims[0].claim_text


def test_extract_statement_claim():
    claims = extract_claims_from_article(
        _article("source-a", "Jorge Jesus said Ronaldo can return.")
    )

    assert claims
    assert claims[0].claim_type == "statement"


def test_selection_and_future_claim_types_are_distinguished():
    selection = extract_claims_from_article(
        _article("source-a", "Portugal vs Norway lineup announced.")
    )
    future = extract_claims_from_article(
        _article(
            "source-b",
            "Cristiano Ronaldo to make Portugal return in 2027.",
        )
    )

    assert selection[0].claim_type == "selection"
    assert future[0].claim_type == "future"


def test_headline_fragment_and_publisher_label_are_not_claim_text():
    fragment = extract_claims_from_article(
        _article(
            "source-a",
            "Norway Lineup Today: Confirmed Team News & Starting XI",
        )
    )
    headline = extract_claims_from_article(
        _article(
            "source-b",
            "Jorge Jesus says Ronaldo can return | Football | Sky Sports",
        )
    )

    assert fragment == []
    assert headline
    assert "Sky Sports" not in headline[0].claim_text


def test_boilerplate_is_not_extracted():
    article = _article(
        "source-a",
        "Read more",
        article_text="Subscribe to our newsletter. Watch live now.",
        snippet="Read more. Subscribe. Watch live.",
    )

    assert extract_claims_from_article(article) == []


def test_noisy_snippet_is_ignored_when_article_text_exists():
    article = _article(
        "source-a",
        "Portugal defeated Norway 2-1.",
        article_text="Portugal defeated Norway 2-1 in the Nations League.",
        snippet="Subscribe now. Arsenal transfer news. Watch live.",
    )

    claims = extract_claims_from_article(article)

    assert all("Arsenal" not in claim.claim_text for claim in claims)


def test_equivalent_result_claims_deduplicate_and_keep_sources():
    first = _article("source-a", "Portugal beat Norway 2-1.", domain="one.example")
    second = _article("source-b", "Portugal defeated Norway 2-1.", domain="two.example")

    groups = extract_claims_from_cluster([first, second])
    result_groups = [group for group in groups if group.claim_type == "result"]

    assert len(result_groups) == 1
    assert set(result_groups[0].source_ids) == {"source-a", "source-b"}


def test_deduplicate_claims_retains_source_evidence():
    groups = deduplicate_claims([_candidate("source-a"), _candidate("source-b")])

    assert len(groups) == 1
    assert groups[0].source_ids == ["source-a", "source-b"]


def test_verifier_marks_conflicting_evidence_disputed():
    assert (
        verify_claim(
            [_candidate("source-a")],
            contradicting_source_count=1,
            confidence=0.9,
        )
        == "disputed"
    )


def test_single_source_is_never_confirmed():
    assert verify_claim([_candidate("source-a")], confidence=0.95) == "developing"


def test_high_priority_source_can_support_a_low_confidence_claim():
    candidate = ClaimCandidate(
        claim_text="Portugal advanced to the next round.",
        source_id="source-a",
        confidence=0.4,
        claim_type="event",
        supporting_text="Portugal advanced to the next round.",
        domain="uefa.com",
        source_priority=100,
    )

    assert verify_claim([candidate], confidence=0.4) == "developing"


def test_pipeline_claims_keep_two_source_evidence_links():
    articles = [
        _article("source-a", "Portugal beat Norway 2-1.", domain="one.example"),
        _article("source-b", "Portugal defeated Norway 2-1.", domain="two.example"),
    ]
    state = cast(
        ResearchState,
        {
            "story_clusters": [
                {
                    "cluster_id": "cluster-1",
                    "representative_article": articles[0],
                    "articles": articles,
                    "semantic_score": 1.0,
                    "independent_source_count": 2,
                    "domains": ["one.example", "two.example"],
                    "created_at": "2026-10-05T11:00:00Z",
                }
            ],
        },
    )

    claims_state = extract_claims(state)
    linked_state = link_evidence(cast(ResearchState, {**state, **claims_state}))
    result_claims = [
        claim for claim in linked_state["claims"] if claim["claim_type"] == "result"
    ]

    assert len(result_claims) == 1
    assert {item["source_id"] for item in result_claims[0]["evidence_sources"]} == {
        "source-a",
        "source-b",
    }
    assert result_claims[0]["evidence_status"] == "developing"


def test_low_quality_article_has_no_candidates():
    article = _article("source-a", "Portugal", "", "Read more. Watch live.")

    assert extract_claims_from_article(article) == []


def test_save_results_persists_claims_and_valid_evidence_links(tmp_path, monkeypatch):
    from football_intelligence import research

    database_path = tmp_path / "claims.sqlite"
    initialize_database(database_path)
    with connect_database(database_path) as connection:
        connection.execute(
            "INSERT INTO research_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("run-test", "Portugal news", "running", "2026-10-05", None, None),
        )

    monkeypatch.setattr(
        research,
        "connect_database",
        lambda: connect_database(database_path),
    )
    article = _article(
        "source-a",
        "Portugal defeated Norway 2-1.",
        article_text="Portugal defeated Norway 2-1 in the Nations League.",
    )
    state = cast(
        ResearchState,
        {
            "query": "Portugal news",
            "run_id": "run-test",
            "max_results": 20,
            "started_at": "2026-10-05",
            "articles": [article],
            "story_clusters": [
                {
                    "cluster_id": "cluster-1",
                    "representative_article": article,
                    "articles": [article],
                    "semantic_score": 1.0,
                    "independent_source_count": 1,
                    "domains": [article["domain"]],
                    "created_at": "2026-10-05T11:00:00Z",
                }
            ],
            "claims": [
                {
                    "story_cluster_id": "cluster-1",
                    "claim_text": "Portugal defeated Norway 2-1.",
                    "claim_type": "result",
                    "confidence": 0.9,
                    "evidence_status": "developing",
                    "evidence_sources": [
                        {
                            "source_id": "source-a",
                            "evidence_type": "supports",
                            "supporting_text": "Portugal defeated Norway 2-1.",
                            "domain": article["domain"],
                            "source_priority": 0,
                        }
                    ],
                }
            ],
            "error": None,
        },
    )

    save_results(state)

    with connect_database(database_path) as connection:
        claim = connection.execute(
            "SELECT claim_id, story_id, claim_text, evidence_status FROM claims"
        ).fetchone()
        evidence = connection.execute(
            "SELECT claim_id, source_id, evidence_type FROM claim_evidence"
        ).fetchone()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert claim["claim_text"] == "Portugal defeated Norway 2-1."
    assert claim["evidence_status"] == "developing"
    assert evidence["claim_id"] == claim["claim_id"]
    assert evidence["source_id"] == "source-a"
    assert evidence["evidence_type"] == "supports"
    assert foreign_key_errors == []
