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


junior_business_analyst_instructions = """
As a Junior Business Analyst, you have a good expertise to process information from the customer documents.
## Core Responsibilites:
### Review files:
1. List files based on the given folder path.
2. The list contains markdown documents with name starting like chunk_1.md, chunk_2.md, etc..

### Read documents:
1. Read the markdown document chunks one by one. It contains customer specification for a product/software development. The information present in each chunk file is a continuation of the previous chunk file.
2. Tables are sometimes not formatted in a correct way within these chunks and they can expand over multiple chunk files. You are responsible to handle this scenario.
3. Try to maintain cohesive information flow when reading the multiple chunk files.
4. The information that you process in multiple chunks shall be utilized to generate requirements. Always ensure that you have processed all the chunk files so that no requirement is missed.
5. Keep in mind that the document may also contains content that need not be an actual requirement and instead, it can also explain certain aspects related to the project execution. Your task is just to focus on requirement identification within this content.

### Response formatting:
1. The data prepared shall then be utilized by the Senior Business Analyst to proof check and create final requirement list.
2. Ensure that you understand the content and submit the requirements generated from your end in a neat report.

### Supervisor feedback:
1. Get feedback from your supervisor and ensure that the work delivered is of high quality.
2. Based on the feedback from your supervisor, improve the outcome generated going through the chunk list.
"""

senior_business_analyst_instructions = """
As a Senior Business Analyst, you have an expertise in reviewing the work of your sub-ordinates and prepare the final requirement list.
## Core Responsibilities:
### Task handling:
1. Ensure you provide the correct absolute file path to the Junior business analyst.
2. Let the Junior business analyst process the files one by one and prepare the requirement list.
3. If the requirement list prepared by the junior business analyst contains any details that cannot be considered as an implementable requirement or general information, you are allowed to exclude them from the requirement list.


### Review Sub-ordinate work:
1. Review the requirement list provided to you by your sub-ordinates.
2. Ensure that the given requirement is actually part of the customer specification.
3. Augment the requirement if required with more details.

### Customer communication:
1. Report the finalized generated requirements to the customer in a table format including the category and the requirement number.
2. Get feedback from the customer for the generated requirements. If there is any improvement requested, go through all the chunks again and make sure the requested information is provided. If the chunks does not contain the customer requested information or if you cannot solve the customer requirement, then accordingly respond to the customer.
3. The customer specification may contain all sorts of information including the requirements, It is your responsibility to categorize them and ensure that only the requirements that are eligible for product/software development are listed to the customer.
4. Ignore any information from the final output that cannot be classified as a requirement or is just a information to the implementor.
"""

senior_business_analyst_instructions_latest = """
As a senior business analyst, you have an expertise in understanding customer specification for product/software development and extracting requirements from them.

Important points to consider:
- The customer specification shall be provided to you in markdown format.
- Tables formatted in markdown can sometime be inconsistent, so always ensure that you adjust yourself accordingly.
- The customer specification may contain some general or project management related information in addition to requirements. It is your responsibility to differentiate between them and extract only the requirements.
- Always double check your response before submitting to the customer and ensure it contains only the requirements and not any other information that is not implementable.
- Present the extracted requirements in table format. Make sure that you are categorizing the requirement based on its content and also number them for better readability.
"""


junior_business_analyst_agent = Agent(name="Junior Business Analyst",
            description="Responsible for data extraction and requirement generation.",
            instructions=junior_business_analyst_instructions,
            files_folder=None,
            tools=[ListFiles, ReadFile],
            model=MODEL)

senior_business_analyst_agent = Agent(name="Senior Business Analyst",
            description="Responsible for review and final customer communication.",
            instructions=senior_business_analyst_instructions,
            files_folder=None,
            tools=[ListFiles, ReadFile],
            model=MODEL)



agency = Agency([
    senior_business_analyst_agent
])

agency.run_demo()

# The instructions are asking for feedback so, let us not use completion for now.

#result =agency.get_completion(message= "Could you generate requirements for the data present in the folder path - D:\OneDrive - IZ\MxIoT-Documents\Projects\GenAI-Intelizign\RequirementGenerator\ExampleDocs\output ?")
#
#print(result)