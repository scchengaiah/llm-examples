import os
from typing import Any, Optional

import psycopg
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads

from src.custom.callbacks.qa_caching_callback_handler import QACachingCallbackHandler

LANGCHAIN_SEMANTIC_CACHING_TABLE = "langchain_semantic_cache_qa"

DB_INIT_SQL = f"""
CREATE TABLE IF NOT EXISTS public.{LANGCHAIN_SEMANTIC_CACHING_TABLE}(
    id SERIAL PRIMARY KEY,
    user_prompt TEXT,
    augmented_prompt TEXT,
    prompt_embedding vector(1536),
    response TEXT,
    metadata JSONB,
    created_at timestamp default current_timestamp
);

CREATE INDEX IF NOT EXISTS langchain_semantic_cache_qa_hnsw_index
    ON public.{LANGCHAIN_SEMANTIC_CACHING_TABLE}
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
    """
    Get the content from the first item after loading the prompt.

    Note that this method may not be required for all invocations.

    :param prompt: A string representing the prompt.
    :return: The content of the first item after loading the prompt.
    """
    return loads(prompt)[0].content


def _init_db():
    with get_postgres_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(DB_INIT_SQL)


"""
Semantic caching customized to handle conversational retrieval QA based scenarios.
"""


class PostgreSQLSemanticCacheQA():

    def __init__(self, embeddings: Embeddings, score_threshold: float = 0.7, top_k: int = 1):
        self.embeddings = embeddings
        self.score_threshold = score_threshold
        self.top_k = top_k
        _init_db()

    def lookup(self, prompt: str) -> Optional[RETURN_VAL_TYPE]:
        with get_postgres_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute((f"""
                    SELECT 
                        user_prompt,
                        augmented_prompt,
                        response,
                        prompt_embedding <=> %(prompt_vector)s::vector as cosine_distance,
                        (1 - (prompt_embedding <=> %(prompt_vector)s::vector)) as cosine_similarity
                    FROM
                        public.{LANGCHAIN_SEMANTIC_CACHING_TABLE}
                    WHERE
                        prompt_embedding is not null
                        and (1 - (prompt_embedding <=> %(prompt_vector)s::vector)) >= %(match_threshold)s
                    ORDER BY 
                        cosine_distance ASC
                    LIMIT %(match_cnt)s
                """), {'prompt_vector': self.embeddings.embed_query(prompt),
                       'match_threshold': self.score_threshold,
                       'match_cnt': self.top_k})
                rows = cursor.fetchall()
                if len(rows) > 0:
                    return [loads(row[2]) for row in rows]

        return None

    def update(self, qa_caching_callback_obj: QACachingCallbackHandler) -> None:
        with get_postgres_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute((f"""
                            INSERT INTO
                                public.{LANGCHAIN_SEMANTIC_CACHING_TABLE}
                            (user_prompt, augmented_prompt, response, prompt_embedding)
                            VALUES
                                (%(user_prompt)s, %(augmented_prompt)s, %(response)s, %(prompt_vector)s::vector)
                        """), {'user_prompt': qa_caching_callback_obj.user_prompt,
                               'augmented_prompt': qa_caching_callback_obj.augmented_prompt,
                               'response': dumps(qa_caching_callback_obj.response),
                               'prompt_vector': self.embeddings.embed_query(qa_caching_callback_obj.user_prompt)})

    def clear(self, **kwargs: Any) -> None:
        with get_postgres_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""TRUNCATE public.%(table_name)s""", {'table_name': LANGCHAIN_SEMANTIC_CACHING_TABLE})
