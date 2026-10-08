from football_intelligence.pdf_report import generate_content_pdf
from football_intelligence.synthesis import (
    ContentResearchPack,
    GroundedText,
    StoryContent,
    synthesize_content,
)


def test_synthesis_references_claims_and_pdf_is_created(tmp_path):
    article = {
        "source_id": "src-1",
        "domain": "reuters.com",
        "publisher": "Reuters",
        "title": "Portugal won the match",
        "published_at": "2026-10-05",
        "url": "https://reuters.com/football/story",
        "article_text": "Portugal won.",
        "search_snippet": "Portugal won.",
    }
    state = {
        "query": "Portugal news",
        "run_id": "test-run",
        "status": "completed",
        "claims": [
            {
                "story_cluster_id": "cluster-1",
                "claim_text": "Portugal won the match",
                "claim_type": "result",
                "confidence": 0.9,
                "evidence_status": "developing",
                "evidence_sources": [
                    {
                        "source_id": "src-1",
                        "supporting_text": "Portugal won.",
                        "evidence_type": "supports",
                    }
                ],
                "contradictions": [],
            }
        ],
        "story_clusters": [
            {
                "cluster_id": "cluster-1",
                "relevance_classification": "primary",
                "representative_article": article,
                "articles": [article],
            }
        ],
    }
    valid = GroundedText(text="Portugal won the match.", claim_indices=[0])
    invalid = GroundedText(text="Portugal scored five goals.", claim_indices=[9])
    unsupported = GroundedText(text="Portugal scored five goals.", claim_indices=[0])
    story = StoryContent(
        cluster_id="cluster-1",
        title="Portugal's result",
        status="confirmed",
        briefing=[valid, unsupported],
        key_facts=[valid, invalid],
    )

    class FakeModel:
        def invoke(self, _prompt):
            return ContentResearchPack(primary_stories=[story])

    pack = synthesize_content(state, model=FakeModel())
    assert len(pack.primary_stories[0].briefing) == 1
    assert len(pack.primary_stories[0].key_facts) == 1
    pdf = generate_content_pdf(state, pack, tmp_path)
    assert pdf.name == "test-run_football_research.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")


def test_pack_schema_has_no_social_content_fields():
    assert set(StoryContent.model_fields) == {
        "cluster_id",
        "title",
        "status",
        "briefing",
        "key_facts",
    }


def test_synthesis_batches_without_global_story_cap():
    state = {"query": "football news", "claims": [], "story_clusters": []}
    for i in range(8):
        article = {
            "source_id": f"src-{i}",
            "title": f"Club {i} appoints manager",
            "url": "https://example.com",
            "publisher": "Example",
        }
        cluster_id = f"cluster-{i}"
        state["claims"].append(
            {
                "story_cluster_id": cluster_id,
                "claim_text": f"Club {i} appoints manager",
                "claim_type": "news",
                "confidence": 0.9,
                "evidence_status": "confirmed",
                "evidence_sources": [],
                "contradictions": [],
            }
        )
        state["story_clusters"].append(
            {
                "cluster_id": cluster_id,
                "relevance_classification": "primary",
                "representative_article": article,
                "articles": [article],
            }
        )

    class FakeModel:
        calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            return ContentResearchPack()

    model = FakeModel()
    pack = synthesize_content(state, model=model)
    assert len(pack.primary_stories) == 8
    assert model.calls == 4
