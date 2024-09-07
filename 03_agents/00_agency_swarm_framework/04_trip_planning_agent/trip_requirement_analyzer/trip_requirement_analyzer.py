from agency_swarm import Agent
from tools.search_internet import SearchInternet
from tools.scrape_and_summarize_website import ScrapeAndSummarizeWebsite

class TripRequirementAnalyzer(Agent):
    def __init__(self, model="gpt-4o"):
        super().__init__(
            name = "Trip Requirement Analyzer",
            model= model,
            description="Responsible to understand the trip requirments and plan the locations.",
            instructions="./instructions.md",
            tools=[SearchInternet, ScrapeAndSummarizeWebsite]
        )