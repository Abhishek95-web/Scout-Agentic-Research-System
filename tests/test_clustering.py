from typing import cast

import numpy as np

from football_intelligence import clustering
from football_intelligence.clustering import (
    calculate_story_score,
    cluster_articles,
    should_cluster,
)
from football_intelligence.state import Article


def test_high_semantic_similarity_clusters():
    assert should_cluster(
        score=0.75,
        semantic=0.75,
        entities=0.0,
    )


def test_strong_entity_overlap_can_support_cluster():
    assert should_cluster(
        score=0.68,
        semantic=0.63,
        entities=0.50,
    )


def test_unrelated_articles_do_not_cluster():
    assert not should_cluster(
        score=0.40,
        semantic=0.40,
        entities=0.0,
    )


def test_story_score_is_bounded():
    score = calculate_story_score(
        semantic=1.0,
        entities=1.0,
        title=1.0,
        time=1.0,
    )

    assert 0.0 <= score <= 1.0


def test_borderline_semantics_need_title_or_time_support():
    assert not should_cluster(
        score=0.66, semantic=0.63, entities=0.5, title=0.1, time=0.2
    )
    assert should_cluster(score=0.66, semantic=0.63, entities=0.5, title=0.4, time=0.2)
    assert not should_cluster(
        score=0.66, semantic=0.63, entities=0.5, title=0.1, time=0.0
    )


def test_strong_shared_entities_and_close_time_can_join_one_event():
    assert should_cluster(
        score=0.56, semantic=0.52, entities=0.75, title=0.35, time=1.0
    )
    assert not should_cluster(
        score=0.40, semantic=0.40, entities=0.75, title=0.35, time=1.0
    )


def test_cluster_comparison_avoids_chain_drift(monkeypatch):
    articles = [
        _article("Portugal vs Norway lineup announced", "2026-09-01T10:00:00Z"),
        _article("Portugal vs Norway confirmed starting XI", "2026-09-01T11:00:00Z"),
        _article("Portugal squad update after the match", "2026-09-01T12:00:00Z"),
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.43589], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(clustering, "embed_texts", lambda _: vectors)
    clusters = cluster_articles(articles)

    assert [len(cluster.articles) for cluster in clusters] == [2, 1]
    assert set(vars(clusters[0])) == {
        "cluster_id",
        "representative_article",
        "articles",
        "semantic_score",
        "independent_source_count",
        "domains",
        "created_at",
    }


def test_related_lineup_articles_cluster(monkeypatch):
    articles = [
        _article("Portugal vs Norway lineup announced", "2026-09-01T10:00:00Z"),
        _article("Portugal vs Norway confirmed starting XI", "2026-09-01T11:00:00Z"),
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.43589]], dtype=np.float32)
    monkeypatch.setattr(clustering, "embed_texts", lambda _: vectors)

    clusters = cluster_articles(articles)

    assert len(clusters) == 1
    assert clusters[0].independent_source_count == 2


def _article(title: str, published_at: str) -> Article:
    return cast(
        Article,
        {
            "title": title,
            "search_snippet": "Portugal team news before the Nations League match",
            "published_at": published_at,
            "domain": f"source-{published_at[11:13]}.example",
        },
    )
