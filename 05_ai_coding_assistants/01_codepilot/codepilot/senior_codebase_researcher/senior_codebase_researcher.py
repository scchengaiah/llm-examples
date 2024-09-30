from agency_swarm import Agent
from codepilot.tools.query_vector_store import QueryVectorStore
from codepilot.tools.read_file import ReadFile
from codepilot.tools.fetch_repo_map import FetchRepoMap


class SeniorCodebaseResearcher(Agent):
    def __init__(self, model="gpt-4o"):
        super().__init__(
            name = "Senior Codebase Researcher",
            model= model,
            description="Expert in researching the codebase based on user queries and generate helpful responses.",
            instructions="./instructions.md",
            tools=[QueryVectorStore, ReadFile, FetchRepoMap]
        )