from football_intelligence.claims import ClaimCandidate, verify_claim
from football_intelligence.provenance import (
    compare_claims,
    group_source_origins,
    infer_source_origin,
    normalize_claim,
)


def source(source_id, domain, **extra):
    return {"source_id": source_id, "domain": domain, **extra}


def test_domain_and_attribution_origin_inference():
    assert infer_source_origin(source("r", "www.reuters.com")).origin == "reuters"
    ap = infer_source_origin(
        source(
            "y",
            "yahoo.com",
            article_text="According to the Associated Press, Portugal won.",
        )
    )
    assert (ap.origin, ap.confidence) == ("associated_press", "high")
    assert infer_source_origin(source("x", "example.org")).confidence == "low"
    official = infer_source_origin(source("u", "uefa.com"))
    assert official.kind == "official"
    assert group_source_origins([source("u", "uefa.com")]).independent_origin_count == 0


def test_aliases_and_origin_grouping():
    assert (
        infer_source_origin(source("a", "x.org", publisher="AP News")).origin
        == "associated_press"
    )
    groups = group_source_origins(
        [
            source("a", "yahoo.com", article_text="According to AP, the team won."),
            source("b", "chosun.com", article_text="AP News reports the team won."),
        ]
    )
    assert groups.publisher_count == 2
    assert groups.independent_origin_count == 1


def test_contradiction_detection_is_narrow():
    assert compare_claims(
        "Ronaldo will return to Portugal", "Ronaldo will not return to Portugal"
    ).is_contradiction
    assert compare_claims(
        "Ronaldo is available", "Ronaldo is unavailable"
    ).is_contradiction
    assert compare_claims("Portugal won", "Portugal lost").is_contradiction
    assert not compare_claims(
        "Portugal beat Norway 2-1", "Portugal reached the quarter-finals"
    ).is_contradiction
    assert not compare_claims(
        "Ronaldo left the squad", "Ronaldo could return to Portugal"
    ).is_contradiction
    assert normalize_claim("Portugal beat Norway 2-1.") != normalize_claim(
        "Portugal defeated Norway 2-1."
    )
    assert not compare_claims(
        "Portugal beat Norway 2-1", "Portugal defeated Norway 2-1"
    ).is_contradiction


def candidate(source_id, origin):
    return ClaimCandidate(
        "Portugal won",
        source_id,
        0.85,
        "result",
        "Portugal won",
        f"{source_id}.com",
        80,
        origin,
        "high",
    )


def test_verifier_uses_origins_conservatively():
    assert verify_claim([candidate("a", "ap")], confidence=0.85) == "developing"
    assert (
        verify_claim([candidate("a", "ap"), candidate("b", "ap")], confidence=0.85)
        == "developing"
    )
    assert (
        verify_claim(
            [candidate("a", "reuters"), candidate("b", "bbc")], confidence=0.85
        )
        == "developing"
    )
    official = ClaimCandidate(
        "Portugal won",
        "uefa",
        0.85,
        "result",
        "Portugal won",
        "uefa.com",
        100,
        "uefa",
        "high",
        "official",
    )
    independent = ClaimCandidate(
        "Portugal won",
        "press",
        0.85,
        "result",
        "Portugal won",
        "reuters.com",
        100,
        "reuters",
        "high",
    )
    assert verify_claim([official, independent], confidence=0.85) == "confirmed"
