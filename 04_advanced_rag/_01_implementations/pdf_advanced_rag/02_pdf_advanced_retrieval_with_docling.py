## References:
## https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb

## To use unstructured, we have to satisfy the following pre-requisites.
## In order for the below code to work, we need poppler utils, have downloaded windows version of the same and added the path in PATH variable.
## Downloaded from - https://github.com/oschwartz10612/poppler-windows

## We need tesseract to be available in PATH variable for the below code to work
## Downloaded from - https://github.com/UB-Mannheim/tesseract/releases

import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
env_loaded = load_dotenv()
print(f"Environment variables loaded: {env_loaded}")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = "./data"
image_dir = "./data/images"
temp_dir = "./temp"
chunk_dir= os.path.join(temp_dir, "chunks")
pdf_file = os.path.join(data_dir, "jio-financial-services-annual-report-2023-2024-small.pdf")

from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# Get elements
pdf_analysis_start_time = time.time()
raw_pdf_elements = partition_pdf(
    filename= pdf_file,
    # Using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=image_dir,
)
print(f"PDF analysis took {time.time() - pdf_analysis_start_time} seconds")
print(raw_pdf_elements)