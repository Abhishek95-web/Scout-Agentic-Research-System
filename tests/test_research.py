from football_intelligence import research
from football_intelligence.research import (
    focused_football_query,
    focused_result_matches_query,
    get_domain,
    has_association_football_context,
    make_source_id,
    normalize_url,
)


def test_normalize_url_removes_tracking_parameters():
    url = "https://www.example.com/story?utm_source=test&fbclid=abc"

    assert normalize_url(url) == "https://www.example.com/story"


def test_normalize_url_removes_fragment():
    url = "https://example.com/story#section"

    assert normalize_url(url) == "https://example.com/story"


def test_normalize_url_removes_trailing_slash():
    url = "https://example.com/story/"

    assert normalize_url(url) == "https://example.com/story"


def test_get_domain_removes_www():
    assert get_domain("https://www.example.com/story") == "example.com"


def test_source_id_is_stable():
    url = "https://example.com/story?utm_source=test"

    assert make_source_id(url) == make_source_id("https://example.com/story")


def test_tavily_search_depth_is_passed_to_provider():
    class Client:
        kwargs = None

        def search(self, **kwargs):
            self.kwargs = kwargs
            return {"results": []}

    client = Client()
    research.search_tavily(client, "Portugal squad", 5, search_depth="advanced")
    assert client.kwargs["search_depth"] == "advanced"


def test_run_research_normalizes_mode(monkeypatch):
    class Graph:
        def invoke(self, state):
            return state

    monkeypatch.setattr(research, "build_research_graph", lambda: Graph())
    assert research.run_research("Portugal", mode="Research")["mode"] == "focused"


def test_focused_country_query_is_disambiguated_as_soccer():
    assert (
        focused_football_query("India Brazil")
        == "India Brazil vs football match soccer"
    )
    assert (
        focused_football_query("Portugal football team news")
        == "Portugal football team news"
    )


def test_focused_mode_requires_association_football_evidence():
    assert has_association_football_context("India Brazil live football match")
    assert not has_association_football_context("Google search profiles open in Brazil")


def test_focused_headlines_must_match_subject_anchors():
    assert focused_result_matches_query(
        "India face Uruguay in friendly", "India Brazil"
    )
    assert not focused_result_matches_query(
        "Messi bids farewell to Argentina", "India Brazil"
    )
