from football_intelligence.quality import (
    assess_quality,
)


def test_short_article_is_not_acceptable():
    result = assess_quality(
        article_text="Too short",
        title="Portugal football",
    )

    assert result.acceptable is False


def test_normal_article_is_acceptable():
    article = "Portugal football news. " * 100

    result = assess_quality(
        article_text=article,
        title="Portugal team news",
    )

    assert result.acceptable is True
    assert result.score > 0
