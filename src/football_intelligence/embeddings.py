from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.
    """

    return SentenceTransformer(MODEL_NAME)


def build_embedding_text(
    title: str,
    snippet: str,
) -> str:
    """
    Build the text representation used
    for semantic embedding.
    """

    return f"Title: {title}\nSummary: {snippet}"


def embed_texts(
    texts: list[str],
) -> np.ndarray:
    """
    Convert texts into normalized embeddings.
    """

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.astype(np.float32)


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Calculate cosine similarity between
    two normalized embedding vectors.
    """

    if first.size == 0 or second.size == 0:
        return 0.0

    return float(np.dot(first, second))
