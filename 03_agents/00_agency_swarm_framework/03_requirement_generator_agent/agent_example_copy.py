import os
from agency_swarm import set_openai_key
from agency_swarm import set_openai_client
from openai import AzureOpenAI
from agency_swarm import BaseTool
from pydantic import Field
from agency_swarm import Agent
from agency_swarm import Agency

from dotenv import load_dotenv
env_loaded = load_dotenv("../.env")
print(f"Env loaded: {env_loaded}")

MODEL = "gpt-4o"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    timeout=60,
    max_retries=5,
)

set_openai_client(client)

class ListFiles(BaseTool):
    """List files based on the given folder path."""

    absolute_folder_path: str = Field(..., description="Absolute path of the folder.")

    # This code will be executed if the agent calls this tool
    def run(self):
        try:
            # List all files and directories in the specified folder
            entries = os.listdir(self.absolute_folder_path)

            # Filter out only files and get their absolute paths
            files = [os.path.join(self.absolute_folder_path, entry) for entry in entries if os.path.isfile(os.path.join(self.absolute_folder_path, entry))]
            
            return files
        except FileNotFoundError:
            print(f"The folder {self.absolute_folder_path} does not exist.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

class ReadFile(BaseTool):
    """Read file based on the absolute file path."""

    absolute_file_path: str = Field(..., description="Absolute path of the file.")

    # This code will be executed if the agent calls this tool
    def run(self):
        try:
            with open(self.absolute_file_path, 'r', encoding='utf-8') as file:
                contents = file.read()
            return contents
        except FileNotFoundError:
            return "File not found."
        except UnicodeDecodeError:
            return "Error decoding file with the provided encoding."
        except Exception as e:
            return f"An error occurred: {e}"


ceo_instructions = """
As a CEO of a successful startup, you are responsible to receive requirements from the reputed customers and deliver the requirements with high quality. 

## Core Responsibilites:
1. Delegate tasks to the appropriate team based on the requirements. Your team has relevant tools to complete a given task.
2. Provide feedback to the team to correct or improve the deliverables.
3. Ensure that the deliverables are matching the customer requirements.
4. Your response to the customer is final and the customer may not be able to provide feedback. Hence, perform a quality check on the deliverables before submitting the final response to the customer.

IF YOU DO NOT HAVE CONTEXT FOR A GIVEN QUESTION/REQUIREMENT. INFORM THE USER ABOUT IT POLITELY. DO NOT HALLUCINATE WITH GENERAL INFORMATION.

"""

senior_business_analyst_instructions = """
As a Senior business analyst of a successful startup, you are highly capable to understand the customer requirements and take appropriate actions to deliver high quality output.

## Backstory

A large document from the customer has been carefully parsed into multiple chunks. All of these chunks are stored inside a folder whose file name start with "chunk_". 

### Important points to note while analyzing these file chunks

1. It should be noted that the images present in the original large document has been converted into a textual form wrapped within <IMAGE_IN_TEXTUAL_REPRESENTATION> xml tags. Each image has Image title and Image description.

2. All the tables present in the original large document are converted into HTML <table> representation to preserve the purity of the information.

## Core Responsibilites:
1. You navigate all the chunk files present in the given folder, understand its context and compare with the given requirements. Prepare your response only after navigating all the chunk files to avoid information loss.
2. Keep the relevant part of the information and ignore unnecessary information that is not relevant to the requirement.
3. The text may contain lot of other additional information that may not be required to satisfy the given requirement. You can ignore such text.
4. It is highly important that you go through all the file chunks, remember the context and prepare your final response for the given requirements.
5. If in case, the consolidated text from all file chunks does not fit in your context window, then accordingly summarize the information from multiple chunks without losing the essential context and prepare the final response.

IF YOU DO NOT HAVE CONTEXT OR KNOW ANSWER TO A QUESTION, INFORM THE USER ABOUT IT POLITELY. DO NOT HALLUCINATE WITH GENERAL INFORMATION.

## Respone format:
1. Format your response in a presentable format.

"""


senior_business_analyst_agent = Agent(name="Senior Business Analyst",
            description="Responsible for analyzing large chunks of data and prepare response based on the customer requirements.",
            instructions=senior_business_analyst_instructions,
            files_folder=None,
            tools=[ListFiles, ReadFile],
            max_completion_tokens=4096,
            model=MODEL)

ceo_agent = Agent(name="CEO",
            description="Responsible for customer communication, task delegation, deliverables review and maintaining quality.",
            instructions=ceo_instructions,
            files_folder=None,
            max_completion_tokens=4096,
            model=MODEL)

agency = Agency([
    ceo_agent,
    [ceo_agent, senior_business_analyst_agent]
])

# Test via Terminal
agency.run_demo()

# Test via Gradio App
# agency.demo_gradio(share=True)

# PRE-PROMPT:
# There is a large document that is chunked into several files residing within the folder D:\tmp\unstructured\pdfImages2\chunks. You will be given a requirement from the customer based on which you traverse through these chunks and deliver relevant response to the customer.# Requirement from the customer:\n 

# The questions are appended to the PRE-PROMPT above during the conversation initiation and for subsequent messages, the requirement shall be directly sent to the Agent without PRE-PROMPT
# QUESTION 1:
# This is a Request for Proposal document and I need you to identify the requirements given as part of this document and list them down. The document may contain lot of other project management and company specific information. You need to focus upon generating the requirements alone ignoring all the other details.

# QUESTION 2:
# Can you explain the responsibility matrix section in detail ?

USER_QUESTION = "Can you explain the responsibility matrix section in detail ?"

CHUNK_LOCATION = "D:/tmp/unstructured/pdfImages2/chunks"

FINAL_PROMPT = f"There is a large document that is chunked into several files residing within the folder {CHUNK_LOCATION}. You will be given a requirement from the customer based on which you traverse through these chunks and deliver relevant response to the customer.# Requirement from the customer:\n{USER_QUESTION}"

# Prepare FINAL_PROMPT for the first message. All follow-up questions can be directly passed to the agent since the
# thread is already initialized.
# result = agency.get_completion(message= FINAL_PROMPT)
# print(result)



