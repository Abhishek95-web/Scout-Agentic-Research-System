from dataclasses import dataclass


@dataclass(frozen=True)
class QualityResult:
    score: float
    content_length: int
    acceptable: bool


def assess_quality(
    article_text: str,
    title: str,
) -> QualityResult:

    content_length = len(article_text.strip())

    score = 0.0

    # Content availability
    if content_length >= 300:
        score += 0.25

    if content_length >= 1000:
        score += 0.25

    if content_length >= 3000:
        score += 0.25

    # Useful title
    if len(title.strip()) >= 20:
        score += 0.15

    # Reasonable article body
    if content_length >= 5000:
        score += 0.10

    score = min(
        1.0,
        score,
    )

    acceptable = content_length >= 300

    return QualityResult(
        score=round(
            score,
            4,
        ),
        content_length=content_length,
        acceptable=acceptable,
    )
