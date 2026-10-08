from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher

from .embeddings import (
    build_embedding_text,
    cosine_similarity,
    embed_texts,
)
from .state import Article


@dataclass
class StoryCluster:
    cluster_id: str
    representative_article: Article
    articles: list[Article]
    semantic_score: float
    independent_source_count: int
    domains: list[str]
    created_at: str


def parse_datetime(
    value: str | None,
) -> datetime | None:
    """
    Parse an ISO-like datetime.

    Returns None if parsing fails.
    """

    if not value:
        return None

    try:
        normalized = value.replace(
            "Z",
            "+00:00",
        )

        parsed = datetime.fromisoformat(normalized)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)

        return parsed

    except ValueError:
        return None


def extract_entities(
    article: Article,
) -> set[str]:
    """
    Extract simple high-value entities from
    an article without requiring an external
    NLP model.

    This is intentionally conservative.
    """

    text = (f"{article['title']} {article['search_snippet']}").lower()

    known_entities = {
        "cristiano ronaldo",
        "ronaldo",
        "portugal",
        "jorge jesus",
        "denmark",
        "rafael leao",
        "rafael leão",
        "norway",
        "uefa",
        "nations league",
        "al nassr",
        "manchester united",
        "real madrid",
        "juventus",
        "premier league",
        "champions league",
        "fifa",
        "lionel messi",
        "messi",
        "argentina",
        "benin",
    }

    return {entity for entity in known_entities if entity in text}


def entity_overlap(
    first: Article,
    second: Article,
) -> float:
    """
    Calculate overlap between important entities.
    """

    first_entities = extract_entities(first)

    second_entities = extract_entities(second)

    if not first_entities or not second_entities:
        return 0.0

    intersection = first_entities & second_entities

    union = first_entities | second_entities

    if not union:
        return 0.0

    return len(intersection) / len(union)


def title_similarity(
    first: Article,
    second: Article,
) -> float:
    """
    Small lexical signal used only as a
    supporting feature.
    """

    return SequenceMatcher(
        None,
        first["title"].lower(),
        second["title"].lower(),
    ).ratio()


def time_similarity(
    first: Article,
    second: Article,
) -> float:
    """
    Give a stronger relationship to articles
    published close together.

    This is a guardrail, not the primary signal.
    """

    first_date = parse_datetime(first.get("published_at"))

    second_date = parse_datetime(second.get("published_at"))

    if not first_date or not second_date:
        return 0.5

    difference_hours = abs((first_date - second_date).total_seconds()) / 3600

    if difference_hours <= 6:
        return 1.0

    if difference_hours <= 24:
        return 0.8

    if difference_hours <= 72:
        return 0.5

    if difference_hours <= 168:
        return 0.2

    return 0.0


def calculate_story_score(
    semantic: float,
    entities: float,
    title: float,
    time: float,
) -> float:
    """
    Combine semantic and supporting signals.

    Semantic similarity is deliberately the
    strongest signal.
    """

    score = semantic * 0.65 + entities * 0.20 + title * 0.10 + time * 0.05

    return round(
        min(1.0, score),
        4,
    )


def should_cluster(
    score: float,
    semantic: float,
    entities: float,
    title: float | None = None,
    time: float | None = None,
) -> bool:
    """
    Decide whether two articles belong to
    the same story.

    Strong semantic similarity can stand alone. In the full pipeline, borderline
    matches also need entity agreement and either title or time support. Optional
    supporting values preserve the original three-argument helper behavior.
    """

    if semantic >= 0.72:
        return True

    # Strong entity agreement plus close publication time can identify one event
    # when outlets phrase their headlines differently (for example a player's
    # final appearance and a report on that same match).
    if (
        semantic >= 0.52
        and entities >= 0.65
        and time is not None
        and time >= 0.80
        and score >= 0.55
    ):
        return True

    if semantic < 0.62 or entities < 0.40:
        return False

    if title is None and time is None:
        return True

    title_support = title is not None and title >= 0.30
    time_support = time is not None and time >= 0.80
    if title_support or time_support:
        return score >= 0.66

    return False


def cluster_articles(
    articles: list[Article],
) -> list[StoryCluster]:
    """
    Group articles into semantic story clusters.
    """

    if not articles:
        return []

    texts = [
        build_embedding_text(
            article["title"],
            article["search_snippet"],
        )
        for article in articles
    ]

    embeddings = embed_texts(texts)
    article_indices = {id(article): index for index, article in enumerate(articles)}

    clusters: list[StoryCluster] = []

    for index, article in enumerate(articles):
        best_cluster = None
        best_score = 0.0

        for cluster in clusters:
            representative_index = article_indices[id(cluster.representative_article)]
            representative_match = False
            member_scores = []

            # Compare with every member, but require agreement with the original
            # representative too so a chain of merely related stories cannot drift.
            for member in cluster.articles:
                member_index = article_indices[id(member)]
                semantic = cosine_similarity(
                    embeddings[index], embeddings[member_index]
                )
                entities = entity_overlap(article, member)
                title = title_similarity(article, member)
                time = time_similarity(article, member)
                score = calculate_story_score(
                    semantic=semantic,
                    entities=entities,
                    title=title,
                    time=time,
                )
                member_scores.append((score, semantic, entities, title, time))
                if member_index == representative_index:
                    representative_match = should_cluster(
                        score, semantic, entities, title, time
                    )

            if representative_match:
                for score, semantic, entities, title, time in member_scores:
                    if (
                        should_cluster(score, semantic, entities, title, time)
                        and score > best_score
                    ):
                        best_score = score
                        best_cluster = cluster

        if best_cluster is None:
            cluster = StoryCluster(
                cluster_id=(f"story-{len(clusters) + 1}"),
                representative_article=article,
                articles=[article],
                semantic_score=1.0,
                independent_source_count=1,
                domains=[article["domain"]],
                created_at=(datetime.now(UTC).isoformat()),
            )

            clusters.append(cluster)

        else:
            best_cluster.articles.append(article)

            if article["domain"] not in (best_cluster.domains):
                best_cluster.domains.append(article["domain"])

            best_cluster.independent_source_count = len(set(best_cluster.domains))

            best_cluster.semantic_score = max(
                best_cluster.semantic_score,
                best_score,
            )

    return clusters
