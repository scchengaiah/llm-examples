from agency_swarm import set_openai_client
from openai import AzureOpenAI
from agency_swarm import Agency, Agent
from codepilot.senior_codebase_researcher.senior_codebase_researcher import SeniorCodebaseResearcher
from textwrap import dedent
import os

from dotenv import load_dotenv
env_loaded = load_dotenv("./.env")
print(f"Env loaded: {env_loaded}")

MODEL = "gpt-4o"

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

senior_codebase_researcher_agent = SeniorCodebaseResearcher(model=MODEL)

agency_manifesto = """# Expert Developer Assistant Agency

You are a part of a reputed organization that assists programmers and software engineers to execute their tasks with ease.

Your mission is to assist them with their tasks and enhance their productivity."""

agency =  Agency(
            agency_chart= [senior_codebase_researcher_agent],
            shared_instructions=agency_manifesto
        )

agency.run_demo()