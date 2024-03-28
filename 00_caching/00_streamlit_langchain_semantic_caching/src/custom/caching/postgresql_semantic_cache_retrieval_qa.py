from typing import Any, Optional, List
import os

import psycopg
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads


DB_INIT_SQL = """
CREATE TABLE IF NOT EXISTS public.langchain_semantic_cache(
    id SERIAL PRIMARY KEY,
    prompt TEXT,
    prompt_embedding vector(1536),
    response TEXT,
    created_at timestamp default current_timestamp
);

CREATE INDEX IF NOT EXISTS langchain_semantic_cache_hnsw_index
    ON public.langchain_semantic_cache
    USING hnsw
    (prompt_embedding vector_cosine_ops)
    WITH (m = 8, ef_construction = 20);
"""



def get_postgres_conn_url():
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = os.getenv("POSTGRES_PORT")
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_db = os.getenv("POSTGRES_DB")

    return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"


def get_postgres_conn():
    return psycopg.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],  # The port you exposed in docker-compose.yml
        dbname=os.environ["POSTGRES_DB"]
    )


def get_prompt_content(prompt: str):
    return loads(prompt)[0].content


def _init_db():
    with get_postgres_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(DB_INIT_SQL)


class PostgreSQLSemanticCache(BaseCache):

    def __init__(self, embeddings: Embeddings, score_threshold: float = 0.7, top_k: int = 1):
        self.embeddings = embeddings
        self.score_threshold = score_threshold
        self.top_k = top_k
        _init_db()

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        prompt_content = get_prompt_content(prompt)
        with get_postgres_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        prompt,
                        response,
                        prompt_embedding <=> %(prompt_vector)s::vector as cosine_distance,
                        (1 - (prompt_embedding <=> %(prompt_vector)s::vector)) as cosine_similarity
                    FROM
                        langchain_semantic_cache
                    WHERE
                        prompt_embedding is not null
                        and (1 - (prompt_embedding <=> %(prompt_vector)s::vector)) >= %(match_threshold)s
                    ORDER BY 
                        cosine_distance ASC
                    LIMIT %(match_cnt)s
                """, {'prompt_vector': self.embeddings.embed_query(prompt_content),
                      'match_threshold': self.score_threshold,
                      'match_cnt': self.top_k})
                rows = cursor.fetchall()
                if len(rows) > 0:
                    return [loads(row[1]) for row in rows]

        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        prompt_content = get_prompt_content(prompt)
        with get_postgres_conn() as conn:
            with conn.cursor() as cursor:
                for idx, gen in enumerate(return_val):
                    cursor.execute("""
                        INSERT INTO
                            langchain_semantic_cache
                        (prompt, response, prompt_embedding)
                        VALUES
                            (%(prompt)s, %(response)s, %(prompt_vector)s::vector)
                    """, {'prompt': prompt_content,
                          'response': dumps(gen),
                          'prompt_vector': self.embeddings.embed_query(prompt_content)})

    def clear(self, **kwargs: Any) -> None:
        with get_postgres_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                        TRUNCATE langchain_semantic_cache
                    """)
