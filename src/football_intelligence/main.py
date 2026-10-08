"""Windows-safe command line entry point for football content research."""

import logging
import sys

from football_intelligence.config import get_settings
from football_intelligence.database import initialize_database
from football_intelligence.pdf_report import generate_content_pdf
from football_intelligence.research import run_research
from football_intelligence.synthesis import synthesize_content

logger = logging.getLogger(__name__)


def _make_console_safe() -> None:
    """Prevent UnicodeEncodeError on Windows cp1252 terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def main() -> None:
    _make_console_safe()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = get_settings()
    initialize_database()
    print(f"\n{settings.app_name}")
    print("Football Research Brief")
    print("-------------------------")
    query = input(
        "Enter a football news query (press Enter for today's news): "
    ).strip()
    query = query or "football news today"
    mode_input = (
        input("Mode: [N]ormal broad news or [F]ocused research? [N]: ").strip().lower()
    )
    mode = "focused" if mode_input in {"f", "focused", "research"} else "normal"
    print("\nCollecting and validating football news...")

    result = run_research(
        query=query, max_results=settings.tavily_max_results, mode=mode
    )
    if result["error"]:
        print(f"Research failed: {result['error']}")
        return

    clusters = result.get("story_clusters", [])
    primary = sum(item["relevance_classification"] == "primary" for item in clusters)
    related = sum(item["relevance_classification"] == "related" for item in clusters)
    print(f"Research run: {result['run_id']}")
    print(f"Articles collected: {len(result['articles'])}")
    print(f"Stories: {len(clusters)} | Primary: {primary} | Related: {related}")
    print(f"Claims: {len(result.get('claims', []))}")
    evidence_links = sum(
        len(item["evidence_sources"]) for item in result.get("claims", [])
    )
    print(f"Evidence links: {evidence_links}")
    print("Preparing factual research brief...")

    try:
        content = synthesize_content(result)
        output_path = generate_content_pdf(result, content)
    except Exception as exc:
        logger.exception("Could not create the content research pack")
        print(f"Content pack generation failed: {type(exc).__name__}: {exc}")
        return

    print("Research brief generated")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
