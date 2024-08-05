import base64
import sys
from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
import os
import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from unstructured.chunking.title import chunk_by_title
import json
import time

# Start time
start_time = time.time()

env_loaded = load_dotenv(".env")
print(f"Env loaded: {env_loaded}")

# In order for the below code to work, we need poppler utils, have downloaded windows version of the same and added the path in PATH variable.
# Downloaded from - https://github.com/oschwartz10612/poppler-windows

# We need tesseract to be available in PATH variable for the below code to work
# Downloaded from - https://github.com/UB-Mannheim/tesseract/releases

FILE_PATH = "D:/gitlab/learnings/artificial-intelligence/llm-examples/03_agents/00_agency_swarm_framework/03_requirement_generator_agent/resources/Scale_NCA_RFP_PLM.pdf"
FILE_PATH = "D:/tmp/appendix_6m_tool_requirement_specifications_v2_project_portfolio_management_tool.pdf"
FILE_PATH = "C:/Users/20092/Documents/Temp/pdftoocr/pngtopdf_2.pdf"

# Unstructured via langchain framework.
def langchain_example():
    loader = UnstructuredPDFLoader(FILE_PATH, 
                                mode="elements",
                                strategy="hi_res")

    docs = loader.load()

    for doc in docs:
        print("**********************************")
        print(f"Category: {doc.metadata['category']}")
        print(f"Page number: {doc.metadata['page_number']}")
        print(doc.page_content)

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
        max_tokens=4096,
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
            max_tokens=4096,
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

# Unstructured via standard library.
# Get elements and also chunk the text blocks - LESS PERFORMANT.

# raw_pdf_elements = partition_pdf(
#     filename=FILE_PATH,
#     
#     # Using pdf format to find embedded image blocks
#     extract_images_in_pdf=True,
#     
#     # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
#     # Titles are any sub-section of the document
#     infer_table_structure=True,
#     
#     # Post processing to aggregate text once we have the title
#     chunking_strategy="by_title",
#     # Chunking params to aggregate text blocks
#     # Attempt to create a new chunk 3800 chars
#     # Attempt to keep chunks > 2000 chars
#     # Hard max on chunks
#     max_characters=4000,
#     new_after_n_chars=3800,
#     combine_text_under_n_chars=2000,
#     image_output_dir_path="static/pdfImages/",
# )

# If some extracted images have content clipped, you can adjust the padding by specifying two environment variables
# os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "20"
os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = "20"

pdf_elements = partition_pdf(filename=FILE_PATH,
                             infer_table_structure=True,
                             extract_images_in_pdf=True,
                             extract_image_block_types=["Image", "Table"],
                             extract_image_block_to_payload=True,
                             # Below argument shall output images to the specified path 
                             # only if extract_image_block_to_payload is set to False 
                             extract_image_block_output_dir="D:/tmp/unstructured/pdfImages4/"
                            )

# Filter elements that carry essential information.
# Convert images to textual representation.
# Convert tables to HTML representation.
# NOTE THAT IMAGES AND TABLES ARE DETECTED AS IMAGES BY THE UNSTRUCTURED LIBRARY OUT OF THE BOX.
ALLOWED_ELEMENTS = ["NarrativeText", "ListItem", "Title", "Address", "EmailAddress", "Image", "Table",
                    "UncategorizedText"]
updated_elements = []
for i, element in enumerate(pdf_elements):
    if element.category in ALLOWED_ELEMENTS:
        if element.category == "Image":
            print(f"Encountered Image. Extract description for the image.")
            image_info = fetch_image_description(element.metadata.image_base64, element.metadata.image_mime_type)
            image_text = f"""<IMAGE_IN_TEXTUAL_REPRESENTATION>
            {image_info}
            </IMAGE_IN_TEXTUAL_REPRESENTATION>
            """
            element.text = '\n'.join(line.strip() for line in image_text.splitlines())
        elif element.category == "Table":
            print(f"Encountered Table. Convert table to HTML representation.")
            html_table_txt = convert_table_to_html(element.metadata.image_base64, element.metadata.image_mime_type)
            table_text = f"""Below is a table extracted from the original document and represented in HTML format. 
            ```html
            {html_table_txt}
            ```
            """
            element.text = '\n'.join(line.strip() for line in table_text.splitlines())
    
        updated_elements.append(element)

chunked_elements = chunk_by_title(
    elements=updated_elements,
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
)

chunk_output_folder = "D:/tmp/unstructured/pdfImages4_ocr/chunks"
os.makedirs(chunk_output_folder, exist_ok=True)

for i, chunk in enumerate(chunked_elements):
    with open(f"{chunk_output_folder}/chunk_{i+1}.txt", "w", encoding="utf-8") as f:
        f.write(chunk.text)

# End time
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f"Time taken: {int(time_taken)} seconds")