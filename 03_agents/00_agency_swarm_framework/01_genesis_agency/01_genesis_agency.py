# https://vrsen.github.io/agency-swarm/quick_start/#use-genesis-agency

# Using genesis agency from code example notebook
# https://github.com/VRSEN/agency-swarm/blob/main/notebooks/genesis_agency.ipynb

# Using azure Openai endpoint with agency swarm framework
# https://github.com/VRSEN/agency-swarm/blob/main/notebooks/azure.ipynb


"""
Genesis agency swarm automates the creation of agent templates to quickly start with the agent setup.

We have used genesis swarm programmatically to validate the usage of Azure OpenAI endpoint. It should be noted that it can be directly invoked from the command line provided we have the OpenAI platform subscription and OPENAI_API_KEY environment variable set.

To invoke from command line:
agency-swarm genesis --openai_key "<OPENAI_API_KEY>"
"""

import os
from openai import AzureOpenAI
from agency_swarm import set_openai_client
from dotenv import load_dotenv
env_loaded = load_dotenv("../.env")
print(f"Env loaded: {env_loaded}")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2024-02-15-preview",
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    timeout=5,
    max_retries=5,
)

set_openai_client(client)

from agency_swarm.agency.genesis import GenesisAgency


test_agency = GenesisAgency(False)

## BREAKING
## This is not working with the Azure OpenAI endpoint since we could not pass the model value to the Agent class through Agency.
test_agency.run_demo()

