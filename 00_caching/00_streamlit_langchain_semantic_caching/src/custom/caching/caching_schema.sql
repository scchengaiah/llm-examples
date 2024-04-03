CREATE TABLE IF NOT EXISTS public.langchain_semantic_cache(
    id SERIAL PRIMARY KEY,
    prompt TEXT,
    prompt_embedding vector(1536),
    response TEXT,
    metadata JSONB,
    created_at timestamp default current_timestamp
);

CREATE INDEX IF NOT EXISTS langchain_semantic_cache_hnsw_index
    ON public.langchain_semantic_cache
    USING hnsw
    (prompt_embedding vector_cosine_ops)
    WITH (m = 8, ef_construction = 20);