from football_intelligence.relevance import (
    assess_relevance,
    calculate_football_score,
    calculate_unrelated_penalty,
    normalize_query_terms,
)


def test_query_terms():
    terms = normalize_query_terms("Manchester City transfer news")

    assert "manchester" in terms
    assert "city" in terms
    assert "transfer" in terms
    assert "news" in terms


def test_relevant_article():
    result = assess_relevance(
        query="Portugal team news",
        title=("Ronaldo leaves Portugal camp after dispute with coach"),
        snippet=("Portugal captain Cristiano Ronaldo left the national team camp."),
    )

    assert result.score > 0.4
    assert result.rejected is False


def test_unrelated_nfl_article():
    result = assess_relevance(
        query="Portugal team news",
        title="NFL Week 4 injury report",
        snippet=("Latest American football injuries and team news."),
    )

    assert result.unrelated_penalty >= 0.8
    assert result.rejected is True


def test_unrelated_tennis_article():
    result = assess_relevance(
        query="Portugal team news",
        title="Tennis star wins final",
        snippet=("Latest tennis results and tournament news."),
    )

    assert result.unrelated_penalty >= 0.8
    assert result.rejected is True


def test_football_detection():
    score = calculate_football_score(
        title="Portugal football team",
        snippet=("National team prepares for the next match."),
    )

    assert score > 0


def test_unrelated_detection():
    penalty = calculate_unrelated_penalty(
        title="NFL Week 4",
        snippet="American football results.",
    )

    assert penalty >= 0.8


def test_american_football_and_fantasy_terms_are_filtered():
    for title in (
        "NCAA football rankings",
        "Fantasy football lineup",
        "College football playoff",
    ):
        assert calculate_unrelated_penalty(title, "") >= 0.8


def test_distinctive_anchor_beats_generic_terms():
    portugal = assess_relevance(
        "Portugal team news", "Portugal prepare for Norway", "A national squad update."
    )
    chelsea = assess_relevance(
        "Portugal team news", "Chelsea team news", "Latest updates."
    )

    assert portugal.score > chelsea.score + 0.3
    assert portugal.score > 0.6


def test_missing_anchor_is_low_and_rejected():
    result = assess_relevance(
        "Portugal team news", "Arsenal announce new squad", "The club made an update."
    )

    assert result.score < 0.2
    assert result.rejected


def test_ronaldo_alias_matches_cristiano_ronaldo():
    result = assess_relevance(
        "Ronaldo future", "Cristiano Ronaldo considers his future", ""
    )

    assert result.score > 0.6
    assert "ronaldo" in result.matched_terms


def test_generic_football_query_remains_usable():
    result = assess_relevance(
        "football news", "Football news: latest match updates", ""
    )

    assert result.score > 0.2
    assert not result.rejected


def test_multiple_anchors_reward_article_with_both():
    both = assess_relevance("Portugal Ronaldo", "Portugal call up Ronaldo", "")
    one = assess_relevance("Portugal Ronaldo", "Portugal prepare for match", "")

    assert both.score > one.score


def test_contextual_portugal_article_is_not_rejected():
    result = assess_relevance(
        "Portugal team news",
        "Portugal coach discusses upcoming squad changes",
        "The national side is preparing for its next match.",
    )

    assert not result.rejected


def test_contextual_ronaldo_article_supports_portugal_query():
    result = assess_relevance(
        "Portugal team news", "Jorge Jesus says Ronaldo can return", ""
    )

    assert result.score > 0.2
    assert "portugal" in result.matched_terms
    assert not result.rejected


def test_unrelated_football_anchor_is_rejected():
    result = assess_relevance(
        "Portugal team news", "Scotland football news", "Latest update."
    )

    assert result.score < 0.1
    assert result.rejected


def test_unrelated_title_is_not_rescued_by_noisy_snippet_anchor():
    result = assess_relevance(
        "Portugal team news",
        "Erling Haaland injury fear after Norway match",
        "Latest ESPN headlines: Portugal squad news, Israel and Arsenal updates.",
    )

    assert result.score < 0.25
    assert result.rejected


def test_generic_query_does_not_require_entity_anchor():
    result = assess_relevance(
        "football news",
        "Manchester City prepare for Premier League clash",
        "",
    )

    assert result.score >= 0.1
    assert not result.rejected
