from agency_swarm import set_openai_client
from openai import AzureOpenAI
from agency_swarm import Agency, Agent
from trip_requirement_analyzer.trip_requirement_analyzer import TripRequirementAnalyzer
from travel_concierge.travel_concierge import TravelConcierge
from textwrap import dedent
import os

from dotenv import load_dotenv
env_loaded = load_dotenv("./.env")
print(f"Env loaded: {env_loaded}")

MODEL = "gpt-4o"

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

trip_requirement_analyzer_agent = TripRequirementAnalyzer(model=MODEL)

travel_concierge_agent = TravelConcierge(model=MODEL)

agency_manifesto = """# "Dream Travel Agency" Agency Manifesto

You are a part of a reputed travel agency.

Your mission is to plan top notch vacations for the reputed customers of our agency."""

agency =  Agency(
            agency_chart= [trip_requirement_analyzer_agent, 
                           [trip_requirement_analyzer_agent, travel_concierge_agent],
                           [travel_concierge_agent, trip_requirement_analyzer_agent]
                           ],
            shared_instructions=agency_manifesto
        )


print("## Welcome to Trip Planning Agent")
print('-------------------------------')

travelling_from = input(
  dedent("""
    From where will you be travelling from?
  """)) or "Nuremberg"

vacation_destination = input(
  dedent("""
    What is your vacation destination?
  """)) or "Switzerland"

date_range = input(
  dedent("""
    What is your date range?. Example format: YYYY-MM-DD - YYYY-MM-DD
  """)) or "2025-05-13 - 2025-05-18"

preferences = input(
  dedent("""
    What are your preferences?
  """)) or "I am travelling with my spouse and 5 year old kid. I have interests in natural attractions and sightseeings. Expecting the trip to be complete, comfortable for my family and budget friendly. Do let me know without fail if I have to book multiple hotels if the distance between sightseeings are too far."

message = f""" Here are the requirements for the trip from the customer:
Travelling from: {travelling_from}
Vacation Destination: {vacation_destination}
Date Range (YYYY-MM-DD - YYYY-MM-DD): {date_range}
Preferences: {preferences}
"""

print(message)

# result = agency.get_completion(message, yield_messages=False)
# print(result)

agency.run_demo()