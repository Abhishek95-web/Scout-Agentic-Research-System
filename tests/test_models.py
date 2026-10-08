import pytest
from pydantic import ValidationError

from football_intelligence.models import (
    Claim,
    EvidenceStatus,
    ResearchRun,
    ResearchStatus,
    Source,
    Story,
)


def test_source_requires_valid_url():
    source = Source(
        url="https://www.example.com/article",
        domain="example.com",
        title="Example football report",
    )

    assert str(source.url) == "https://www.example.com/article"


def test_source_rejects_invalid_url():
    with pytest.raises(ValidationError):
        Source(
            url="not-a-url",
            domain="example.com",
            title="Invalid source",
        )


def test_claim_defaults_to_unverified():
    claim = Claim(text="A transfer has been reported.")

    assert claim.evidence_status == EvidenceStatus.UNVERIFIED
    assert claim.supporting_source_ids == []
    assert claim.contradicting_source_ids == []


def test_story_importance_score_must_be_in_range():
    with pytest.raises(ValidationError):
        Story(
            title="Test story",
            importance_score=101,
        )


def test_research_run_has_pending_status_by_default():
    run = ResearchRun(
        run_id="run-test-001",
        query="football news today",
    )

    assert run.status == ResearchStatus.PENDING
    assert run.completed_at is None


def test_story_can_hold_claims_and_sources():
    claim = Claim(
        text="The club announced a new signing.",
        evidence_status=EvidenceStatus.CONFIRMED,
        supporting_source_ids=["source-1"],
    )

    story = Story(
        title="Club announces signing",
        claims=[claim],
        source_ids=["source-1"],
    )

    assert len(story.claims) == 1
    assert story.claims[0].supporting_source_ids == ["source-1"]
