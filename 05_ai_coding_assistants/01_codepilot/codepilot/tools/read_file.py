from agency_swarm import BaseTool
from pydantic import Field
import os

BASE_PATH = "D:/tmp/genai/agency-swarm"

class ReadFile(BaseTool):
    """Read file based on the relative file path."""

    relative_file_path: str = Field(..., description="Relative file path of the file.")

    # This code will be executed if the agent calls this tool
    def run(self):
        try:
            absolute_file_path = os.path.join(BASE_PATH, self.relative_file_path)
            with open(absolute_file_path, 'r', encoding='utf-8') as file:
                contents = file.read()
            return contents
        except FileNotFoundError:
            return "File not found."
        except UnicodeDecodeError:
            return "Error decoding file with the provided encoding."
        except Exception as e:
            return f"An error occurred: {e}"