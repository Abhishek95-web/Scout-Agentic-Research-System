"""Professional, concise PDF rendering for football research briefs."""

from __future__ import annotations

import html
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

from football_intelligence.state import ResearchState
from football_intelligence.synthesis import ContentResearchPack, StoryContent


def _safe(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return html.escape(normalized, quote=True)


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 11 * mm, "Football Research Brief")
    canvas.drawRightString(192 * mm, 11 * mm, f"{document.page}")
    canvas.restoreState()


def generate_content_pdf(
    state: ResearchState, pack: ContentResearchPack, output_dir: Path | None = None
) -> Path:
    destination = output_dir or Path(__file__).resolve().parents[2] / "data" / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{state['run_id']}_football_research.pdf"
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleBlue",
            parent=styles["Title"],
            textColor=colors.HexColor("#12304A"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="StoryTitle",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#087E8B"),
            spaceBefore=12,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallGrey",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#526174"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            keepWithNext=True,
            spaceAfter=6,
        )
    )
    clusters = {item["cluster_id"]: item for item in state.get("story_clusters", [])}
    document_sources: set[str] = set()
    flow = [
        Paragraph("FOOTBALL RESEARCH BRIEF", styles["TitleBlue"]),
        Paragraph(_safe(state["query"]), styles["Heading2"]),
        Paragraph(
            f"Mode: {_safe(state.get('mode', 'normal')).title()} &nbsp; | &nbsp; {_safe(datetime.now(UTC).date().isoformat())}",
            styles["SmallGrey"],
        ),
        Spacer(1, 5 * mm),
    ]

    def add_story(story: StoryContent) -> None:
        cluster = clusters.get(story.cluster_id, {})
        story_flow = [Paragraph(_safe(story.title), styles["StoryTitle"])]
        story_flow.append(
            Paragraph(
                f"Status: <b>{_safe(story.status.title())}</b>", styles["SmallGrey"]
            )
        )
        seen_facts: set[str] = set()
        for item in story.briefing:
            key = item.text.strip().casefold()
            if key in seen_facts:
                continue
            seen_facts.add(key)
            story_flow.append(Paragraph(_safe(item.text), styles["BodyText"]))
        if story.key_facts:
            for item in story.key_facts:
                key = item.text.strip().casefold()
                if key in seen_facts:
                    continue
                seen_facts.add(key)
                story_flow.append(
                    Paragraph(f"&#8226; {_safe(item.text)}", styles["BodyText"])
                )
        seen: set[str] = set()
        links = []
        for article in cluster.get("articles", []):
            url = str(article.get("url", ""))
            if url and url not in seen and url not in document_sources:
                seen.add(url)
                document_sources.add(url)
                links.append(
                    (article.get("publisher", "Source"), article.get("title", url), url)
                )
        if links:
            story_flow.append(Paragraph("Sources", styles["SmallGrey"]))
            for publisher, title, url in links[:6]:
                story_flow.append(
                    Paragraph(
                        f'{_safe(publisher)} — <link href="{_safe(url)}" color="#087E8B">{_safe(title)}</link>',
                        styles["SmallGrey"],
                    )
                )
        flow.append(KeepTogether(story_flow))

    for heading, stories in (
        ("Top stories", pack.primary_stories),
        ("Related developments", pack.related_stories),
    ):
        if stories:
            flow.append(Paragraph(heading, styles["Section"]))
            for story in stories:
                add_story(story)
    if not pack.primary_stories and not pack.related_stories:
        flow.append(
            Paragraph(
                "No sufficiently supported stories were found for this query.",
                styles["BodyText"],
            )
        )
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Football Research Brief",
        author="Football Intelligence",
    )
    document.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    return path
