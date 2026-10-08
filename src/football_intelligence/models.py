from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceStatus(StrEnum):
    CONFIRMED = "confirmed"
    DEVELOPING = "developing"
    DISPUTED = "disputed"
    UNVERIFIED = "unverified"


class SourceType(StrEnum):
    OFFICIAL = "official"
    NEWS_OUTLET = "news_outlet"
    JOURNALIST = "journalist"
    OTHER = "other"


class ResearchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    domain: str = Field(min_length=1)
    title: str = Field(min_length=1)
    publisher: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    source_type: SourceType = SourceType.OTHER
    article_text: str = ""
    content_hash: str | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    supporting_source_ids: list[str] = Field(default_factory=list)
    contradicting_source_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class Story(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    summary: str = ""
    topic: str = "football"
    teams: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    importance_score: float = Field(default=0.0, ge=0.0, le=100.0)


class ResearchRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    status: ResearchStatus = ResearchStatus.PENDING
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    story_ids: list[str] = Field(default_factory=list)
    error_message: str | None = None
