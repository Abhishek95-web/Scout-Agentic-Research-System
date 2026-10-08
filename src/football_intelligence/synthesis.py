"""Concise factual synthesis grounded in the existing claim and evidence graph."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from football_intelligence.config import get_settings
from football_intelligence.state import ResearchState

logger = logging.getLogger(__name__)
_FALLBACK_MODEL = "openai/gpt-oss-120b"
_NOISE_CLAIM = re.compile(
    r"\b(nfl|nba|mlb|nhl|ncaa|cfp|waiver wire|fantasy football|"
    r"leave your rankings in the comments|post your picks|"
    r"instagram\s+youtube\s+x\s+spotify|thefantasytool|fantasytool|fantasy tool|"
    r"you have permission to edit|you have reached espn|"
    r"jewelers|lab-grown diamonds|newsroom login)\b",
    re.IGNORECASE,
)


class GroundedText(BaseModel):
    text: str = Field(min_length=1)
    claim_indices: list[int] = Field(min_length=1, max_length=4)


class StoryContent(BaseModel):
    cluster_id: str
    title: str
    status: Literal["confirmed", "developing", "disputed", "unverified"]
    briefing: list[GroundedText] = Field(default_factory=list, max_length=2)
    key_facts: list[GroundedText] = Field(default_factory=list, max_length=4)


class ContentResearchPack(BaseModel):
    primary_stories: list[StoryContent] = Field(default_factory=list)
    related_stories: list[StoryContent] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list)


class StoryBatch(BaseModel):
    """Batch response schema without run-level metadata."""

    primary_stories: list[StoryContent] = Field(default_factory=list)
    related_stories: list[StoryContent] = Field(default_factory=list)


def _story_inputs(state: ResearchState) -> list[dict]:
    claims_by_cluster: dict[str, list[dict]] = {}
    for index, claim in enumerate(state.get("claims", [])):
        if _NOISE_CLAIM.search(claim["claim_text"]):
            continue
        claims_by_cluster.setdefault(claim["story_cluster_id"], []).append(
            {
                "index": index,
                "claim": claim["claim_text"],
                "status": claim["evidence_status"],
                "confidence": claim["confidence"],
            }
        )
    inputs = []
    for cluster in state.get("story_clusters", []):
        if cluster.get("relevance_classification") == "low_relevance":
            continue
        claims = sorted(
            claims_by_cluster.get(cluster["cluster_id"], []),
            key=lambda c: (c["status"] == "unverified", -c["confidence"]),
        )
        if not claims:
            continue
        statuses = {claim["status"] for claim in claims}
        status = next(
            (s for s in ("disputed", "confirmed", "developing") if s in statuses),
            "unverified",
        )
        inputs.append(
            {
                "cluster_id": cluster["cluster_id"],
                "classification": cluster.get("relevance_classification", "primary"),
                "title": cluster["representative_article"]["title"],
                "status": status,
                "claims": claims,
            }
        )
    inputs.sort(
        key=lambda item: (item["classification"] != "primary", -len(item["claims"]))
    )
    return inputs


def synthesize_content(
    state: ResearchState, model: object | None = None
) -> ContentResearchPack:
    """Synthesize all relevant stories in small batches, preserving claim references."""
    settings = get_settings()
    configured = settings.groq_model

    def create_model(name: str):
        return ChatGroq(
            model=name,
            api_key=settings.groq_api_key.get_secret_value(),
            temperature=0,
            max_tokens=3000,
        ).with_structured_output(StoryBatch, method="json_schema", strict=True)

    if model is None:
        if settings.groq_api_key is None:
            raise RuntimeError(
                "GROQ_API_KEY is required to synthesize the research brief"
            )
        model = create_model(configured)

    stories = _story_inputs(state)
    by_id = {item["cluster_id"]: item for item in stories}
    claim_rows = state.get("claims", [])
    generated: list[StoryContent] = []
    # Keep prompts bounded without dropping stories from the research run.
    for start in range(0, len(stories), 2):
        batch = stories[start : start + 2]
        payload = {
            "query": state["query"],
            "stories": batch,
            "rules": [
                "Use only supplied verified claim text.",
                "Every briefing/fact must cite supplied claim indices.",
                "Do not invent details or upgrade evidence status.",
                "Return factual football news only; no content or posting suggestions.",
            ],
        }
        prompt = (
            "Write a concise factual news brief for each supplied story. Include one short briefing and up to four key facts, each grounded in claim indices.\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        try:
            result = model.invoke(prompt)  # type: ignore[attr-defined]
        except Exception as exc:
            if (
                model is not None
                and settings.groq_api_key is not None
                and configured != _FALLBACK_MODEL
                and getattr(exc, "status_code", None) == 404
            ):
                logger.warning(
                    "Configured Groq model unavailable; retrying with %s",
                    _FALLBACK_MODEL,
                )
                result = create_model(_FALLBACK_MODEL).invoke(prompt)
            else:
                raise
        parsed = (
            StoryBatch.model_validate(result) if isinstance(result, dict) else result
        )
        candidates = parsed.primary_stories + parsed.related_stories
        for story in candidates:
            raw = by_id.get(story.cluster_id)
            if raw is None or story.cluster_id not in {
                item["cluster_id"] for item in batch
            }:
                continue
            allowed = {item["index"] for item in raw["claims"]}

            def valid(line: GroundedText, allowed_indices: set[int] = allowed) -> bool:
                if (
                    not line.claim_indices
                    or not set(line.claim_indices) <= allowed_indices
                ):
                    return False
                cited = " ".join(
                    claim_rows[i]["claim_text"] for i in line.claim_indices
                ).lower()
                return all(
                    token in cited
                    for token in __import__("re").findall(
                        r"[a-z0-9]+", line.text.lower()
                    )
                    if len(token) > 4
                    and token
                    not in {
                        "about",
                        "after",
                        "their",
                        "there",
                        "which",
                        "where",
                        "while",
                        "reported",
                        "sources",
                    }
                )

            story.title = raw["title"]
            story.status = raw["status"]
            story.briefing = [line for line in story.briefing if valid(line)][:2]
            story.key_facts = [line for line in story.key_facts if valid(line)][:4]
            if not story.briefing:
                story.briefing = [
                    GroundedText(
                        text=raw["claims"][0]["claim"],
                        claim_indices=[raw["claims"][0]["index"]],
                    )
                ]
            if not story.key_facts:
                story.key_facts = [
                    GroundedText(text=item["claim"], claim_indices=[item["index"]])
                    for item in raw["claims"][:4]
                ]
            generated.append(story)
    # Deterministic completeness even when the model omits a story.
    generated_ids = {item.cluster_id for item in generated}
    for raw in stories:
        if raw["cluster_id"] not in generated_ids:
            lines = [
                GroundedText(text=c["claim"], claim_indices=[c["index"]])
                for c in raw["claims"][:4]
            ]
            generated.append(
                StoryContent(
                    cluster_id=raw["cluster_id"],
                    title=raw["title"],
                    status=raw["status"],
                    briefing=lines[:1],
                    key_facts=lines,
                )
            )
    primary = [
        s for s in generated if by_id[s.cluster_id]["classification"] == "primary"
    ]
    related = [
        s for s in generated if by_id[s.cluster_id]["classification"] != "primary"
    ]
    notes = [
        f"{claim['claim_text']} — {claim['evidence_status']}"
        for claim in claim_rows
        if claim["evidence_status"] in {"disputed", "unverified"}
    ]
    return ContentResearchPack(
        primary_stories=primary, related_stories=related, verification_notes=notes
    )
