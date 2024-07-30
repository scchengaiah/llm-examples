import os
from openai import AzureOpenAI
from agency_swarm import set_openai_client
import logging

logging.basicConfig(level=logging.INFO)

MODEL = "gpt-4o"

from dotenv import load_dotenv
env_loaded = load_dotenv("../.env")
print(f"Env loaded: {env_loaded}")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    timeout=5,
    max_retries=5,
)

set_openai_client(client)

from agency_swarm import Agent
ceo = Agent(name="ceo", description="I am the CEO", model=MODEL)
agent1 = Agent(name="agent1", description="I am a simple agent", model=MODEL)

from agency_swarm import Agency

agency = Agency([ceo, [ceo, agent1]])

## BREAKING
# Not working with the latest updates in the OpenAI Assistant API (V2).
# Downgraded agency-swarm to 0.1.7 to test the implementation.
response = agency.get_completion("Say hi to agent1. Let me know his response.", yield_messages=False)
print(response)

#agency.run_demo()