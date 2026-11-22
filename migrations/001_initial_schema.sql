-- 001_initial_schema.sql
-- ResearchOS schema for runs, events, and evaluation history

CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR(64) PRIMARY KEY,
    question TEXT NOT NULL,
    depth VARCHAR(20) NOT NULL DEFAULT 'standard',
    max_papers INT NOT NULL DEFAULT 6,
    status VARCHAR(30) NOT NULL DEFAULT 'pending', -- pending, running, completed, failed
    summary_md TEXT,
    reference_list JSONB DEFAULT '[]'::jsonb,
    claims JSONB DEFAULT '[]'::jsonb,
    comparison JSONB DEFAULT '{}'::jsonb,
    eval_scores JSONB DEFAULT '{}'::jsonb,
    trace JSONB DEFAULT '[]'::jsonb,
    timings JSONB DEFAULT '{}'::jsonb,
    call_counts JSONB DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);

CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_run_id_id ON events (run_id, id ASC);

CREATE TABLE IF NOT EXISTS eval_reports (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64),
    dataset_name VARCHAR(100) NOT NULL DEFAULT 'default_8_qa',
    retrieval_ablation JSONB NOT NULL DEFAULT '{}'::jsonb,
    factual_consistency JSONB NOT NULL DEFAULT '{}'::jsonb,
    detailed_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eval_reports_created_at ON eval_reports (created_at DESC);
