import sqlite3
from pathlib import Path

from football_intelligence.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error_message TEXT
);


CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,

    url TEXT UNIQUE NOT NULL,

    domain TEXT NOT NULL,

    title TEXT,

    publisher TEXT,

    author TEXT,

    published_at TEXT,

    retrieved_at TEXT NOT NULL,

    source_type TEXT NOT NULL,

    -- Search/source strategy metadata
    source_tier TEXT NOT NULL DEFAULT 'standard',

    source_priority INTEGER NOT NULL DEFAULT 0,

    source_category TEXT NOT NULL DEFAULT 'unknown',

    article_text TEXT,

    content_hash TEXT
);


CREATE TABLE IF NOT EXISTS stories (
    story_id TEXT PRIMARY KEY,

    run_id TEXT NOT NULL,

    title TEXT NOT NULL,

    summary TEXT,

    topic TEXT,

    evidence_status TEXT NOT NULL,

    importance_score REAL,

    created_at TEXT NOT NULL,

    FOREIGN KEY (run_id)
        REFERENCES research_runs(run_id)
);


CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,

    story_id TEXT NOT NULL,

    claim_text TEXT NOT NULL,

    evidence_status TEXT NOT NULL,

    notes TEXT,

    FOREIGN KEY (story_id)
        REFERENCES stories(story_id)
);


CREATE TABLE IF NOT EXISTS story_sources (
    story_id TEXT NOT NULL,

    source_id TEXT NOT NULL,

    PRIMARY KEY (
        story_id,
        source_id
    ),

    FOREIGN KEY (story_id)
        REFERENCES stories(story_id),

    FOREIGN KEY (source_id)
        REFERENCES sources(source_id)
);


CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id TEXT NOT NULL,

    source_id TEXT NOT NULL,

    evidence_type TEXT NOT NULL,

    PRIMARY KEY (
        claim_id,
        source_id
    ),

    FOREIGN KEY (claim_id)
        REFERENCES claims(claim_id),

    FOREIGN KEY (source_id)
        REFERENCES sources(source_id)
);
"""


def connect_database(path: Path | None = None) -> sqlite3.Connection:
    settings = get_settings()
    database_path = path or settings.database_path

    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    return connection


def initialize_database(path: Path | None = None) -> None:
    with connect_database(path) as connection:
        connection.executescript(SCHEMA)
