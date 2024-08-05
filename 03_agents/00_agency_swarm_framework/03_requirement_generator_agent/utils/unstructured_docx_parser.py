import base64
import json
import os
import sys
import time

import instructor
from anthropic import AnthropicBedrock
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from pydantic import BaseModel, Field, field_validator
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.docx import partition_docx

# Start time
start_time = time.time()

env_loaded = load_dotenv(".env")
print(f"Env loaded: {env_loaded}")

# In order for the below code to work, we need poppler utils, have downloaded windows version of the same and added the path in PATH variable.
# Downloaded from - https://github.com/oschwartz10612/poppler-windows

# We need tesseract to be available in PATH variable for the below code to work
# Downloaded from - https://github.com/UB-Mannheim/tesseract/releases

FILE_PATH = "D:/gitlab/learnings/artificial-intelligence/llm-examples/03_agents/00_agency_swarm_framework/03_requirement_generator_agent/resources/appendix_6m_tool_requirement_specifications_v2_project_portfolio_management_tool.docx"

HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
SONNET_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
instructor_client = instructor.from_anthropic(
                                    AnthropicBedrock(
                                        aws_access_key=os.getenv("AWS_ACCESS_KEY"),
                                        aws_secret_key=os.getenv("AWS_SECRET_KEY"),
                                        aws_region=os.getenv("AWS_REGION"),
                                    )
                                )
anthropic_client = AnthropicBedrock(
                        aws_access_key=os.getenv("AWS_ACCESS_KEY"),
                        aws_secret_key=os.getenv("AWS_SECRET_KEY"),
                        aws_region=os.getenv("AWS_REGION"),
                    )

def image_to_base64(image_path):
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Read the image file
        image_data = image_file.read()
        # Encode the image data in base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded

def fetch_image_description(b64_image_data, media_type="image/jpeg"):
    resp = anthropic_client.messages.create(
        model=HAIKU_MODEL_ID,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You shall be provided with an image. Try to understand its content and provide a detailed description of the image along with a suitable title. Try to focus on the content present in the image and not the aesthetics of the image. \n## Response format:\nFormat your response as below:\nImage Title:\nImage Description:\n"
                    }
                ],
            }
        ]
    )

    print(resp.content[0].text)
    return resp.content[0].text

def convert_table_to_html(b64_image_data, media_type="image/jpeg"):
    resp = anthropic_client.messages.create(
            model=SONNET_MODEL_ID,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "You shall be provided with an image that resembles a table. Convert the image representation of the table to HTML table. The output should be in HTML format including just the <table> tag. Do not include any other details in the output other than the HTML <table> representation."
                        }
                    ],
                }
            ]
        )

    print(resp.content[0].text)
    return resp.content[0].text



docx_elements = partition_docx(filename=FILE_PATH,
                             infer_table_structure=True,
                             strategy="hi_res"
                            )

# End time
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f"Time taken: {int(time_taken)} seconds")