## References:
## https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb

## To use unstructured, we have to satisfy the following pre-requisites.
## In order for the below code to work, we need poppler utils, have downloaded windows version of the same and added the path in PATH variable.
## Downloaded from - https://github.com/oschwartz10612/poppler-windows

## We need tesseract to be available in PATH variable for the below code to work
## Downloaded from - https://github.com/UB-Mannheim/tesseract/releases

import os
from typing import List
from dotenv import load_dotenv
from pathlib import Path
import time
import logging

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_cohere import CohereRerank
from langchain.storage import LocalFileStore
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
import uuid
import psycopg
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


def get_title(markdown_text):
    class TitleResponse(BaseModel):
        title: str = Field(description="First level header title extracted for the provided markdown text.")
    
    prompt = (
        "You are an expert at extracting concise, meaningful title from markdown text. "
        "Generate a first-level header title for the provided markdown text. "
        "The title should reflect the provided content without any deviation. "
        "Keep the title concise and do not add too much of information to it.\n\n"
        "Markdown text: \n\n"
        f"{markdown_text}\n\n"
        "Title:\n"
    )

    messages = HumanMessage(
        content=[
            {"type": "text", "text": prompt}
        ],
    )
    
    response = azure_openai_model.with_structured_output(TitleResponse).invoke([messages])
    
    return response.title

def clean_query_for_bm25(query):
    # Escape single quotes by doubling them for PostgreSQL
    # Example: "JFSL's" becomes "JFSL''s"
    return query.replace("'", "''")

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

                

            child_docs.extend(modified_splits)

    ## for i, doc in enumerate(child_docs):
    ##     print("-" * 80)
    ##     print(f"Split {i}:\n\n Metadata: \n{doc.metadata}\n\n Content: \n{doc.page_content}\n\n")
    return parent_docs, child_docs

def init_pg_vectorstore(recreate_collection=False):
    db_host = "172.31.60.199"
    db_user = "admin"
    db_password = "admin"
    db_name = "postgres"
    db_port = "25432"
    # Database connection parameters
    db_params = {
        "dbname": db_name,
        "user": db_user,
        "password": db_password,
        "host": db_host,  # Use the appropriate host
        "port": db_port  # Default PostgreSQL port
    }

    # Connect to the PostgreSQL database
    # with psycopg.connect(**db_params) as conn:
    #     print("Postgresql Test connection successful.")

    connection = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    collection_name = "pdf_visualizer_test"

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small", api_version="2024-02-01")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    if recreate_collection:
        # Drop Collection if existing.
        vectorstore.delete_collection()

        # Create Empty collection
        vectorstore.create_collection()

    return vectorstore

def ingest_to_vectorstore_with_multi_vector_retriever(recreate_collection=True):
    byte_store = LocalFileStore(os.path.join(temp_dir, "byte-store"))
    id_key = "doc_id"
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore = init_pg_vectorstore(recreate_collection),
        byte_store=byte_store,
        id_key=id_key,
    )
    
    # Ingest all the chunks into the vector store
    print(f"Number of documents to ingest into the vector store: {len(child_docs)}")
    retriever.vectorstore.add_documents(child_docs, ids=[doc.metadata['id'] for doc in child_docs])

    # Ingest all the chunks into the byte store
    doc_ids = [doc.metadata['id'] for doc in parent_docs]
    # Access the doc store through retriever since it takes care of serialization and de-serialization.
    retriever.docstore.mset(list(zip(doc_ids, parent_docs)))
    print("Documents ingested into the vector store and doc store successfully.")

def multi_vector_hybrid_retrieval(input_queries):
    ## We use hardcoded SQL query on langchain_pg_embedding table. For actual implementation, create customized implementation
    ## extending the langchain abstractions for standard usage.
    sql = """
    WITH semantic_search AS (
        SELECT id, RANK () OVER (ORDER BY embedding <=> %(embedding)s::vector) AS rank
        FROM langchain_pg_embedding ORDER BY embedding <=> %(embedding)s::vector LIMIT 50
    ),
    bm25_search AS (
        SELECT id, RANK () OVER (ORDER BY paradedb.score(id) DESC) as rank
        FROM langchain_pg_embedding WHERE document @@@ %(query)s LIMIT 50
    )
    SELECT
        COALESCE(semantic_search.id, bm25_search.id) AS id,
        COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
        COALESCE(1.0 / (%(k)s + bm25_search.rank), 0.0) AS score,
        langchain_pg_embedding.document,
        langchain_pg_embedding.embedding,
        langchain_pg_embedding.cmetadata
    FROM semantic_search
    FULL OUTER JOIN bm25_search ON semantic_search.id = bm25_search.id
    JOIN langchain_pg_embedding ON langchain_pg_embedding.id = COALESCE(semantic_search.id, bm25_search.id)
    ORDER BY score DESC, document
    LIMIT 20;
    """

    consolidated_docs = []
    # We do this to maintain the order of the ids that are returned
    ids = []

    for input_query in input_queries:
        _log.info(f"Processing query: {input_query}")
        embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small", api_version="2024-02-01")
        query_embeddings = embeddings.embed_query(input_query)
        # Use a smaller value of k when you want to give more weight to the keyword search (BM25) results.
        # Try setting k between 5 and 20. This keeps keyword relevance as the dominant factor but still allows semantic scores to contribute.

        # Use a larger value of k to give more weight to the semantic search results.
        # Set k between 60 and 100. Higher values make scores less sensitive to BM25 rank differences, letting the semantic search results have more influence.

        # A value of k between 30 and 60 is typically ideal for balancing BM25 and semantic search
        # Lower end (30-40): Slightly leans toward keyword relevance if that’s your preference.
        # Higher end (50-60): Slightly favors semantic similarity but retains substantial keyword influence.
        # Starting Value: k=50 is often a solid middle-ground starting point. This lets you achieve a reasonable balance, where both ranking lists contribute, yet neither dominates.
        k = 60
        db_host = "172.31.60.199"
        db_user = "admin"
        db_password = "admin"
        db_name = "postgres"
        db_port = "25432"
        conn = psycopg.connect(dbname=db_name, 
                                user=db_user, password=db_password, host=db_host, port=db_port,
                                autocommit=True)
        # Clean the query for BM25 search
        cleaned_query = clean_query_for_bm25(input_query)
        results = conn.execute(sql, {'query': cleaned_query, 'embedding': query_embeddings, 'k': k}).fetchall()

        reranked_docs = []
        print(f"No of rows returned: {len(results)}")

        for row in results:
            reranked_docs.append(Document(page_content=row[2], metadata=row[4]))

        for d in reranked_docs:
            if multi_vector_retriever.id_key in d.metadata and d.metadata[multi_vector_retriever.id_key] not in ids:
                ids.append(d.metadata[multi_vector_retriever.id_key])
    
    consolidated_docs = multi_vector_retriever.docstore.mget(ids)
    
    return [d for d in consolidated_docs if d is not None]

def query_expansion_with_multi_vector_hybrid_retrieval():
    class Questions(BaseModel):
        """Generated Questions for the input query"""
        questions: List[str] = Field(description="Generated questions")

    system_prompt = """You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_query}")
    ])
    messages = prompt.format_messages(user_query=user_query)
    model_id = "gpt-4o-mini"
    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)
    response = azure_openai_model.with_structured_output(Questions, method="json_schema").invoke(messages)
    docs = multi_vector_hybrid_retrieval(input_queries=[user_query] + response.questions)
    return docs

def invoke_llm_rag_with_textual_context():
    prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don’t know the answer, just say that you cannot answer the question due to insufficient context without providing any additional information. "
    "Try to keep the answer concise and elaborate only if the question demands it.\n\n"
    "Question: {question}\n\n"
    "Context: \n{context}\n\n"
    "Answer:\n")

    formatted_context = "\n\n".join(
    [doc.page_content for i, doc in enumerate(docs)])

    messages = prompt.format_messages(question=user_query, context=formatted_context)
    print("*" * 20)
    print("LLM INPUT:")
    print(messages[0].content)
    print("*" * 20)
    model_id = "gpt-4o-mini"
    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)
    ### response = azure_openai_model.invoke(messages)
    ### print("*" * 20)
    ### print("Response:")
    ### print(response.content)
    for chunk in azure_openai_model.stream(messages):
        print(chunk.content, end="", flush=True)

### markdown_export_per_page()

# Parent docs hold the content of the entire page.
# Child docs hold chunked content for all the pages.
### parent_docs, child_docs = prepare_docs_from_chunks()
### ingest_to_vectorstore_with_multi_vector_retriever(recreate_collection=True)

user_query = "who are the board of directors ? Explain about them."

multi_vector_retriever = MultiVectorRetriever(
        vectorstore=init_pg_vectorstore(recreate_collection=False),
        byte_store = LocalFileStore(os.path.join(temp_dir, "byte-store")),
        id_key="doc_id",
        search_type="similarity",
        # For the overall retriever, gets reflected for multi_vector_retriever.invoke(query) method.
        # score_threshold is only valid if search_type="similarity_score_threshold"
        search_kwargs= {"k": 25, "score_threshold": 0.5}
)

# docs = multi_vector_retriever.invoke(user_query)
docs = multi_vector_hybrid_retrieval([user_query])
# docs = query_expansion_with_multi_vector_hybrid_retrieval()

for i, doc in enumerate(docs):
    print("-" * 80)
    print(f"\n\nSplit {i}:\n\nMetadata:\n{doc.metadata}\n\nContent:\n{doc.page_content.strip()}\n\n")


print("Using CohereRerank - START")
rerank_start_time = time.time()
compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=3)
docs = compressor.compress_documents(docs, user_query)
print("Using CohereRerank - END")
print(f"Time taken to rerank documents: {time.time() - rerank_start_time} seconds")

for i, doc in enumerate(docs):
    print("-" * 80)
    print(f"\n\nSplit {i}:\n\nMetadata:\n{doc.metadata}\n\nContent:\n{doc.page_content.strip()}\n\n")

invoke_llm_rag_with_textual_context()

