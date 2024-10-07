from agency_swarm import set_openai_client
from openai import AzureOpenAI
from agency_swarm import Agency, Agent
from codepilot.senior_codebase_researcher.senior_codebase_researcher import SeniorCodebaseResearcher
from textwrap import dedent
from typing import List
from pydantic import BaseModel
import os
from agency_swarm import set_openai_key

from dotenv import load_dotenv
env_loaded = load_dotenv("./.env")
print(f"Env loaded: {env_loaded}")

MODEL = "gpt-4o"
# Use the below model with OPENAI API.
# MODEL = "gpt-4o-2024-08-06"

class VectorStoreResult(BaseModel):
    file_path:str

class AgentResponse(BaseModel):
    agent_response: str
    vector_store_results: List[VectorStoreResult]


client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version=os.getenv("AZURE_API_VERSION"),
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    timeout=120,
    max_retries=5,
)

set_openai_client(client)
# Use the below logic with OPENAI API.
# set_openai_key(os.getenv("OPENAI_API_KEY"))

senior_codebase_researcher_agent = SeniorCodebaseResearcher(model=MODEL)

agency_manifesto = """# Expert Developer Assistant Agency

You are a part of a reputed organization that assists programmers and software engineers to execute their tasks with ease.

Your mission is to assist them with their tasks and enhance their productivity."""

agency =  Agency(
            agency_chart= [senior_codebase_researcher_agent],
            shared_instructions=agency_manifesto
        )

# agency.run_demo()
# result = agency.get_completion(message= "how do I create a custom tool?")
result = agency.get_completion_stream(message= "how do I create a custom tool?")
print(result)