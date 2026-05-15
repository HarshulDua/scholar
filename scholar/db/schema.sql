CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    update_date DATE,
    citation_count INTEGER DEFAULT 0,
    embedding vector(384)
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_history (
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,
    clicked_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, paper_id)
);

CREATE TABLE IF NOT EXISTS click_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT,
    paper_id TEXT,
    query TEXT,
    rank_position INTEGER,
    clicked_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS papers_embedding_idx ON papers USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS user_history_user_idx ON user_history (user_id);
