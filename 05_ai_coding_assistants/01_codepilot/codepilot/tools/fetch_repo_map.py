from agency_swarm import BaseTool
from pydantic import Field
import os

BASE_PATH = "D:/tmp/genai/agency-swarm"
FILE_NAME = "repo_map.txt"

class FetchRepoMap(BaseTool):
    """Fetch repository map of the codebase. Use this tool to get holisitic high level view of the codebase."""

    # This code will be executed if the agent calls this tool
    def run(self):
        try:
            absolute_file_path = os.path.join(BASE_PATH, FILE_NAME)
            with open(absolute_file_path, 'r', encoding='utf-8') as file:
                contents = file.read()
            return contents
        except FileNotFoundError:
            return "Repository map not found."
        except UnicodeDecodeError:
            return "Error decoding file with the provided encoding."
        except Exception as e:
            return f"An error occurred: {e}"