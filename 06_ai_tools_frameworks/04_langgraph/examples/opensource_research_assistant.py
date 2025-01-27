import itertools
import json
import re
import sys
from textwrap import dedent
from typing import Dict, List

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Initialize LLM
llm = ChatOllama(
    base_url="https://cloudsetup-mv.intelizign.com/ollama",
    model="deepseek-r1:14b",
    # model="qwen2.5:14b",
    temperature=0,
    client_kwargs={"verify": False},
)


def remove_think_blocks(text: str) -> str:
    """
    Removes all content within <think> tags (including the tags themselves)
    while preserving the rest of the text exactly as-is.

    Args:
        text: Input string containing XML tags and other content

    Returns:
        str: Original text with <think> blocks removed
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_json_content(text: str) -> str:
    """
    Processes text to:
    1. Remove all content within <think> tags (including the tags)
    2. Extract JSON content from ```json code blocks

    Args:
        text: Input string containing XML and markdown code blocks

    Returns:
        str: Cleaned JSON content as string
    """
    # Remove all content within <think> tags using non-greedy match
    text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Extract JSON content from first ```json block
    json_match = re.search(r"```json(.*?)```", text_without_think, flags=re.DOTALL)

    if not json_match:
        raise ValueError("No JSON code block found in input text")

    return json_match.group(1).strip()


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class SearchQueries(BaseModel):
    search_queries: list[SearchQuery] = Field(
        None,
        description="Comprehensive list of search queries for efficient web search.",
    )


class ContextIndex(BaseModel):
    context_idx_list: list[int] = Field(
        None, description="Context index that is relevant to the user query."
    )


def generate_search_query(query: str) -> list[SearchQuery]:
    """
    Generates optimized web search query from user question
    """
    system_message_instructions = """You are a search query expert. Analyze the user's question and generate an effective web search query 
    that captures the core intent and essential keywords. If the query is complex, you can split them up into multiple sub-queries.

    Generate a maximum of 3 sub-queries.

    ## Response format:
    Ensure you return the output in JSON enclosed with ```json``` tags. Do not include any other text other than the JSON. The JSON should conform to the below provided schema.
    {json_schema}
    """

    system_message = SystemMessage(
        content=system_message_instructions.format(
            json_schema=SearchQueries.model_json_schema()
        )
    )

    human_message = HumanMessage(content=f"User question:\n{query}")

    messages = [system_message] + [human_message]

    response = llm.invoke(messages)

    parsed_response = extract_json_content(response.content)

    search_queries = SearchQueries(**json.loads(parsed_response))

    return search_queries.search_queries


def rank_contexts(query: str, contexts: List[Dict], k: int = 5) -> List[int]:
    """
    Ranks and returns indices of top k most relevant contexts
    """
    context_str = "\n\n".join(
        [f"Index {i}:\n {ctx['content'][:500]}" for i, ctx in enumerate(contexts)]
    )

    system_instructions = """You are an expert reranker who can analyze the provided contexts and rank them by relevance to the user query. 
    
    ## Important instructions:
    1. Return ONLY the top {k} indices.
    2. If multiple indices has same contextual content, then include only one of them.
    3. **Keep in mind that you have to return only the indices of the provided contexts that are relevant to the given user query and not any additional answer or the information.**
    
    ## Response format:
    Ensure you return the output in JSON enclosed with ```json``` tags. Do not include any other text other than the JSON. The JSON should conform to the below provided schema.
    {json_schema}
    """

    system_message = SystemMessage(
        content=dedent(
            system_instructions.format(
                k=str(k), json_schema=ContextIndex.model_json_schema()
            )
        )
    )

    human_message = HumanMessage(
        content=dedent(
            f"User Query: {query}\n\nContexts:\n\n{context_str}\n\nProvide me the top {k} indices of the contexts that are relevant to answer the User query in JSON format. Refer to the JSON schema and ensure that the generated JSON is conforming to the provided schema."
        )
    )

    messages = [system_message] + [human_message]

    response = llm.invoke(messages)

    parsed_response = extract_json_content(response.content)

    relevant_context = ContextIndex(**json.loads(parsed_response))

    return relevant_context.context_idx_list


def generate_response(query: str, contexts: List[Dict]) -> str:
    """
    Generates final answer with citations from ranked contexts
    """
    context_str = "\n\n".join(
        [
            f"[Source {i+1}]: {ctx['content']}\nURL: {ctx['url']}"
            for i, ctx in enumerate(contexts)
        ]
    )

    system_instructions = """You are a research assistant. Using the provided sources, answer the query. 
    - Base your response ONLY on the provided sources.
    - Cite sources using [number] notation.
    - Include a "References" section listing all used sources with URLs.
    - If information isn't available, state that clearly.
    - Format your response in well structured Markdown format.
    """

    system_message = SystemMessage(content=dedent(system_instructions))
    human_message = human_message = HumanMessage(
        content=dedent(f"Query: {query}\n\nSources:\n{context_str}\n\nAnswer:")
    )
    messages = [system_message] + [human_message]
    response = llm.invoke(messages)
    parsed_content = remove_think_blocks(response.content)
    return parsed_content


def research_assistant(query: str, k: int = 5) -> str:
    """
    End-to-end research pipeline
    """
    # Generate search query
    search_queries = generate_search_query(query)

    # Web search (returns list of {'title', 'snippet', 'link'})
    search_tool = DuckDuckGoSearchResults(max_results=3, output_format="list")
    contexts = []
    # Get all search results first
    all_search_results = [
        search_tool.invoke(query.search_query) for query in search_queries
    ]

    # Flatten the list of search results and create contexts
    for search_result in itertools.chain.from_iterable(all_search_results):
        contexts.append(
            {
                "content": f"{search_result['title']}: {search_result['snippet']}",
                "url": search_result["link"],
            }
        )

    # Rank contexts
    context_idx_list = rank_contexts(query, contexts, k)
    top_contexts = [contexts[i] for i in context_idx_list]

    # Generate final response
    return generate_response(query, top_contexts)


# Example usage
if __name__ == "__main__":
    query = "What are the latest advancements in renewable energy storage technologies?"
    query = "Can you research on the process involved to book tickets to kedarnath yatra via helicopter ? Guide me with step by step # instructions and ensure that you ground your research from the trusted sources"
    query = "Research on the latest qwen 2.5 model that supports 1M context window."
    print(research_assistant(query))
