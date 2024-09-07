from agency_swarm import Agent
from tools.search_internet import SearchInternet
from tools.scrape_and_summarize_website import ScrapeAndSummarizeWebsite
from tools.calculator import CalculatorTool

class TravelConcierge(Agent):
    def __init__(self, model="gpt-4o"):
        super().__init__(
            name = "Amazing Travel Concierge",
            model= model,
            description="Specialist in travel planning and logistics with decades of experience.",
            instructions="./instructions.md",
            tools=[SearchInternet, ScrapeAndSummarizeWebsite, CalculatorTool]
        )