from football_intelligence.sources import (
    get_source_category,
    get_source_policy,
    get_source_priority,
    get_source_tier,
    normalize_domain,
)


def test_normalize_domain():
    assert normalize_domain("www.BBC.com") == "bbc.com"


def test_priority_source():
    assert get_source_tier("bbc.com") == "priority"


def test_priority_value():
    assert get_source_priority("reuters.com") == 100


def test_source_category():
    assert get_source_category("transfermarkt.com") == "transfers"


def test_unknown_source():
    assert get_source_tier("randomfootballsite.example") == "standard"

    assert get_source_priority("randomfootballsite.example") == 0


def test_subdomain_matching():
    policy = get_source_policy("sport.bbc.com")

    assert policy is not None

    assert policy.domain == "bbc.com"
