## References:
## https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb

## To use unstructured, we have to satisfy the following pre-requisites.
## In order for the below code to work, we need poppler utils, have downloaded windows version of the same and added the path in PATH variable.
## Downloaded from - https://github.com/oschwartz10612/poppler-windows

## We need tesseract to be available in PATH variable for the below code to work
## Downloaded from - https://github.com/UB-Mannheim/tesseract/releases

import os
from dotenv import load_dotenv
from pathlib import Path
import time
import logging

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import uuid
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
env_loaded = load_dotenv()
_log.info(f"Environment variables loaded: {env_loaded}")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = "./data"
image_dir = "./data/images"
temp_dir = "./temp"
chunk_dir= os.path.join(temp_dir, "chunks")
pdf_file = os.path.join(data_dir, "jio-financial-services-annual-report-2023-2024-small.pdf")

headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4")
    ]

openai_model_id = "gpt-4o-mini"
azure_openai_model = AzureChatOpenAI(model=openai_model_id, max_tokens=8192)

def markdown_export_per_page():
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(chunk_dir)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.ocr_options.use_gpu = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(pdf_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    for page_no, page in conv_res.document.pages.items():
        _log.info(f"Processing Page number: {page_no}")
        content_md = conv_res.document.export_to_markdown(page_no=page_no, image_placeholder="")
        md_filename = output_dir / f"page-chunk-output-{page_no}.md"
        with md_filename.open("wb") as fp:
            fp.write(content_md.encode("utf-8"))

    end_time = time.time() - start_time

    _log.info(f"Document converted and written to markdown per page in {end_time:.2f} seconds.")

# markdown_export_per_page()

def get_title(markdown_text):
    class TitleResponse(BaseModel):
        title: str = Field(description="First level header title extracted for the provided markdown text.")
    
    prompt = (
        "You are an expert at extracting concise, meaningful title from markdown text. "
        "Generate a first-level header title for the provided markdown text. "
        "The title should reflect the provided content without any deviation. \n"
        "Markdown text: \n"
        f"{markdown_text}\n"
        "Title:\n"
    )

    messages = HumanMessage(
        content=[
            {"type": "text", "text": prompt}
        ],
    )
    
    response = azure_openai_model.with_structured_output(TitleResponse).invoke([messages])
    
    return response.title
    

def prepare_docs_from_chunks():

    # Each item corresponds to a page of markdown content.
    parent_docs = []
    # Final docs corresponds to the total chunks for all the pages.
    child_docs = []

    id_key = "doc_id"

    for i, _ in enumerate(os.listdir(chunk_dir)):
        # Increment by one to start from 1
        idx = i+1
        markdown_content = ""
        filename = f"page-chunk-output-{idx}.md"
        file_path = os.path.join(chunk_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                _log.info(f"Processing file:{file_path}")
                markdown_text = file.read().decode('utf-8')
                
            if markdown_text.strip() == "":
                _log.info(f"Skipping empty file: {file_path}")
                continue
            
            # We generate title for the markdown text that acts as a contextualized heading for the content.
            # This step improves the retrieval accuracy.
            content_title = get_title(markdown_text)
            markdown_content += f"# {content_title}\n\n"
            markdown_content += markdown_text

            # The entire markdown_content should be part of the parent chunk.
            # Split the markdown_content into chunks of 1000 characters that should act as child chunks.
            # During retrieval, the parent chunk should be identified based on the child chunk and shall be passed as context to the LLM.
            # This is the concept of Multi-vector Retriever https://python.langchain.com/docs/how_to/multi_vector/
            parent_doc_metadata = {
                "id" : str(uuid.uuid4()),
                "type": "MARKDOWN",
                "file_name": os.path.basename(pdf_file),
                "file_path": pdf_file,
                "page_number": idx,
                "image_path": os.path.join(image_dir, f"page-{idx}.png")
            }
            parent_docs.append(Document(page_content=markdown_content, metadata=parent_doc_metadata))

            # MD splits
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, return_each_line=False, strip_headers=True
            )
            md_header_splits = markdown_splitter.split_text(markdown_content)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )

            splits = text_splitter.split_documents(md_header_splits)

            # First, create a new list to store the modified documents that includes headers as part of the content 
            # and additional metadata.
            modified_splits = []

            # Iterate through the splits and modify each document to include headers as part of the content. This will give more context to the content
            # as indicated here - https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb.
            # This modified_splits should be considered for chunking purpose as it provides more context to the content based on the headings.
            for doc in splits:
                # Create the header string by combining all headers in order
                header_content = ""
                for markdown_symbol, metadata_key in headers_to_split_on:
                    if metadata_key in doc.metadata:
                        header_content += f"{markdown_symbol} {doc.metadata[metadata_key]}\n"
                
                # Create a new document with headers prepended to the content
                new_content = header_content + doc.page_content
                
                # Create a new document with the same metadata but modified content
                # Add type attribute to the metadata.
                doc.metadata["id"] = str(uuid.uuid4())
                doc.metadata[id_key] = parent_doc_metadata["id"] # Add parent doc id
                doc.metadata["type"] = "MARKDOWN"
                doc.metadata["file_name"] = os.path.basename(pdf_file)
                doc.metadata["file_path"] = pdf_file
                doc.metadata["page_number"] = idx
                doc.metadata["image_path"] = os.path.join(image_dir, f"page-{idx}.png")
                modified_doc = Document(page_content=new_content, metadata=doc.metadata)
                modified_splits.append(modified_doc)

            for i, doc in enumerate(modified_splits):
                print("-" * 80)
                print(f"Split {i}:\n\n Metadata: \n{doc.metadata}\n\n Content: \n{doc.page_content}\n\n")

            child_docs.extend(modified_splits)
    return parent_docs, child_docs

# Parent docs hold the content of the entire page.
# Child docs hold chunked content for all the pages.
parent_docs, child_docs = prepare_docs_from_chunks()

