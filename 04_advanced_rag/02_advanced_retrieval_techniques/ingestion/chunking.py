from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from config import settings
from parser.pdf.pdf_parser import PDFParser


def chunk_text(text: str) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"], chunk_size=500, chunk_overlap=0
    )
    text_split = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50,
        tokens_per_chunk=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
        model_name=settings.EMBEDDING_MODEL_ID,
    )
    chunks = []

    for section in text_split:
        chunks.extend(token_splitter.split_text(section))

    return chunks

def chunk_markdown_text(text: str) -> list[str]:
    character_splitter = MarkdownTextSplitter(
        chunk_size=500, chunk_overlap=0
    )
    text_split = character_splitter.split_text(text)
    
    return text_split

def chunk_pdf(pdf_file_path: str) -> list[str]:
    with open(pdf_file_path, "rb") as pdf_file_obj:
        pdf_parser = PDFParser(pdf_file_obj, describe_images=False)
        markdown_document = pdf_parser.parse()

    return chunk_markdown_text(markdown_document)