from football_intelligence.query_intent import classify_story_relevance


def test_generic_football_query_uses_relevance():
    assert (
        classify_story_relevance(
            "football news today", "Portugal announce squad", relevance_score=0.8
        ).classification
        == "primary"
    )
    assert (
        classify_story_relevance(
            "football news today", "Brazil election", relevance_score=0.05
        ).classification
        == "low_relevance"
    )


def test_query_anchors_prefer_direct_headline_match():
    result = classify_story_relevance(
        "India Brazil friendly",
        "India face Brazil in international friendly",
        "Brazil team arrive in India",
        relevance_score=0.8,
    )
    assert result.classification == "primary"


def test_india_uruguay_follow_up_is_related_and_election_is_low():
    follow_up = classify_story_relevance(
        "India Brazil friendly",
        "India vs Uruguay tickets reduced after poor turnout",
        "The decision follows low attendance at the India Brazil friendly.",
        relevance_score=0.7,
    )
    election = classify_story_relevance(
        "India Brazil friendly",
        "Brazil election enters final week",
        "Political parties campaign in Brazil.",
        relevance_score=0.7,
    )
    assert follow_up.classification == "related"
    assert election.classification == "low_relevance"


def test_vinicius_story_tied_to_india_brazil_match_is_primary():
    result = classify_story_relevance(
        "India Brazil friendly",
        "Vinicius shines in India friendly",
        "The Brazil forward played in the India friendly.",
        relevance_score=0.75,
    )
    assert result.classification == "primary"
