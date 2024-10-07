from agency_swarm.tools import BaseTool
from typing import List
from pydantic import Field
import requests
import json
import hashlib
from rich import print as rprint
from textwrap import dedent

BASE_URL = "http://172.22.123.84:8000/api/search"

def generate_md5_result_hash(result: dict) -> str:
    # Generate MD5 hash of the result
    result_str = f"{result['name']}-{result['code_type']}-{result['signature']}-{result['line']}-{result['line_from']}-{result['line_to']}"
    md5_hash = hashlib.md5(result_str.encode('utf-8')).hexdigest()
    return md5_hash

def remove_duplicates(results):
    seen = set()
    unique_results = []
    
    for result in results:
        md5_hash = result['md5_hash']
        if md5_hash not in seen:
            seen.add(md5_hash)
            unique_results.append(result)
    
    return unique_results

def parse_vector_store_response(results):
    if len(results) == 0:
        return "No results retrieved from the vector store."
    
    parsed_results = []
    for result in results:
        parsed_result = f"Filepath: {result['context']['file_path']}\n"
        parsed_result = parsed_result + f"CodeSnippet:\n```python\n{result['context']['snippet']}\n```"
        parsed_results.append(dedent(parsed_result))

    return "\n\n*****************************************************\n\n".join(parsed_results)

class QueryVectorStore(BaseTool):
    """
    Tool for querying a vector store and return semantically matching results for the user provided query.
    """
    user_queries: List[str] = Field(..., description="List of user queries that shall be queried against the vector store.")

    def run(self):
        results = []
        try:
            for query in self.user_queries:
                url_encoded_query = requests.utils.quote(query)
                response = requests.get(f"{BASE_URL}?query={url_encoded_query}")
                result_arr = response.json().get("result")
                for result in result_arr:
                    result["md5_hash"] = generate_md5_result_hash(result)
                    results.append(result)
            unique_results = remove_duplicates(results)
            return parse_vector_store_response(unique_results)
        except Exception as e:
            return f"Exception encountered when querying the vector store: {str(e)}"

if __name__ == "__main__":
    query_vector_store = QueryVectorStore(user_queries=["how to stream response ?", "how to stream response ?"])
    results = query_vector_store.run()
    rprint(results)