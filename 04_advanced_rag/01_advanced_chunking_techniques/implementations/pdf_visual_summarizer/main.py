## Activities: 

### Data Ingestion:
# 1. Read PDF by page and convert each page into an image for LLM Summarization. (Detailed Summary with multiple chunks.)
# 2. Summarize each image using LLM.
# 3. Store the summarized content in vector store with appropriate metadata (Id, Filename, Filepath, PageNumber, ImagePath)

### Advanced Retrieval:
# 1. Create multiple versions of the user query to retrieve relevant context.
# 2. Query the vector store for the user query along with the generated queries and retrieve the top K chunks.
# 3. Rerank - Multiple options
#    Rerank the retrieved chunks based on their relevance to the question using Reciprocal Rank Fusion.
#    Rerank leveraging encoder based models.
#    Rerank using LLM.
#    Contextual embedding + BM25 search

### Response Generation
### Option 1:
# 1. Send the retrieved context to the LLM with appropriate prompt to answer the question.
### Option 2 (Advanced):
# 1. Extract images for the reranked chunks.
# 2. Send the retrieved context along with the image to the LLM with appropriate prompt to answer the question.

from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, Field
import base64
import os
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_postgres.vectorstores import PGVector
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
import pickle
import time
import uuid
import psycopg
from itertools import chain

env_loaded = load_dotenv()

print(f"Env loaded: {env_loaded}")

pdf_file = "./data/jio-financial-services-annual-report-2023-2024-small.pdf"
images_dir = "./data/images"
temp_dir = "./temp"


############################################## Data Ingestion - START ####################################################

########################################################################################################################

#### 1. Read PDF by page and convert each page into an image for LLM Summarization. (Detailed Summary with multiple chunks.)

def convert_pdf_to_images():
    import fitz

    os.makedirs(images_dir, exist_ok=True)

    doc = fitz.open(pdf_file)

    for page_number in range(len(doc)):  # Iterate through the pages
        page = doc.load_page(page_number)  # Load the page
        pix = page.get_pixmap(dpi=150)  # Render the page to a pixmap
        pix.save(f'./data/images/page-{page_number + 1}.png')  # Save the pixmap as a PNG file

    doc.close()


# convert_pdf_to_images()


########################################################################################################################


########################################################################################################################

#### 2. Summarize each image using LLM.

class PropositionalChunk(BaseModel):
    """Propositional Chunk created from the image."""
    image_path: str = Field(description="Path to the image.")
    chunk_number: str = Field(description="Sequence number given to the chunk.")
    propositional_chunk: str = Field(description="Propositional chunk from the image.")


class PropositionalChunks(BaseModel):
    """Propositional Chunk created from the image."""
    image_path: str = Field(description="Path to the image for which the propositional chunks were created.")
    """Consolidated Propositional Chunks created from the image."""
    propositional_chunks: List[PropositionalChunk] = Field(description="List of Propositional chunks from the image.")


def convert_images_to_propositional_chunks():
    model_id = "gpt-4o"

    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)

    image_summarization_prompt_multiple_chunks = """
    You are an expert in understanding the provided image and convert it into a detailed textual representation. Make sure that you capture all aspects of the image and write detailed description of the image content.  Ensure that the textual representation of the image is as comprehensive as possible.

    Image path in the file system: {image_path}

    Format your detailed response into multiple chunks with each chunk holding 200-250 words. Each chunk should be in the form of proposition that should stand on its own. These chunks shall be stored in a vector database to perform a similarity search so prepare it accordingly.
    """

    # Iterate through the images and generate descriptions.
    image_description = []
    batch_messages = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)

        print("Processing image:", image_path)

        # Read the image file and encode it to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Summarize the image using LLM
        messages = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
                },
                {"type": "text", "text": image_summarization_prompt_multiple_chunks.format(image_path=image_path)},
            ],
        )
        batch_messages.append([messages])

    batch_response = azure_openai_model.with_structured_output(PropositionalChunks).batch(batch_messages,
                                                                                          config={"max_concurrency": 5})

    for response in batch_response:
        image_description.append({
            "image_path": response.image_path,
            "propositional_chunks": response.propositional_chunks
        })
    # Serialize the list to a file
    pickle_file = f"{temp_dir}/image_description_{time.time_ns()}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(image_description, file)

    return pickle_file


# image_description_pickle_file = convert_images_to_propositional_chunks()
# print(f"Image description pickle file created: {image_description_pickle_file}")

########################################################################################################################


########################################################################################################################

#### 3. Store the summarized content in vector store with appropriate metadata (Id, Filename, Filepath, PageNumber, ImagePath)

# Deserialize the list from the file
file_path = f"{temp_dir}/image_description_1730829480476374600.pkl"


def print_chunks():
    for image_description in loaded_image_description:
        print("*" * 30)
        print("Image path:", image_description["image_path"])
        for chunk in image_description["propositional_chunks"]:
            print("*" * 20)
            print("Chunk number:", chunk.chunk_number)
            print("Image path:", chunk.image_path)
            print("Chunk content:")
            print(chunk.propositional_chunk)


def prepare_docs_to_ingest():
    metadata_template = {
        "id": None,
        "file_name": None,
        "file_path": None,
        "page_number": None,
        "image_path": None
    }
    docs = []
    for image_description in loaded_image_description:
        for chunk in image_description["propositional_chunks"]:
            image_path = chunk.image_path
            page_number = os.path.basename(image_path).split('-')[1].split('.')[0]
            metadata = metadata_template.copy()
            metadata["id"] = str(uuid.uuid4())
            metadata["file_name"] = os.path.basename(pdf_file)
            metadata["file_path"] = pdf_file
            metadata["page_number"] = page_number
            metadata["image_path"] = image_path
            # Instantiate Document Object and append to the list
            docs.append(Document(page_content=chunk.propositional_chunk, metadata=metadata))
    return docs


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
    with psycopg.connect(**db_params) as conn:
        print("Postgresql Test connection successful.")

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


def ingest_to_vectorstore(recreate_collection=False):
    # Prepare langchain documents to ingest into the pgvector store.
    docs = prepare_docs_to_ingest()
    vectorstore = init_pg_vectorstore(recreate_collection)
    print(f"Number of documents to ingest into the vector store: {len(docs)}")
    vectorstore.add_documents(docs, ids=[doc.metadata['id'] for doc in docs])
    print("Documents ingested into the vector store successfully.")


with open(file_path, 'rb') as file:
    loaded_image_description = pickle.load(file)


# print_chunks()
# ingest_to_vectorstore(recreate_collection = True)

########################################################################################################################

############################################## Data Ingestion - END ####################################################

# ----------------------------------------------------------------------------------------------------------------------

############################################## Advanced Retrieval - START ####################################################

########################################################################################################################

#### 1. Create multiple versions of the user query to retrieve relevant context.
class Questions(BaseModel):
    """Generated Questions for the input query"""
    questions: List[str] = Field(description="Generated questions")


def generate_multiple_queries(user_query):
    system_prompt = """You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_query}")
    ])
    messages = prompt.format_messages(user_query=user_query)
    model_id = "gpt-4o"
    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)
    response = azure_openai_model.with_structured_output(Questions, method="json_schema").invoke(messages)
    return response.questions


# user_query = "what are the services offered by JFSL ?"
user_query = "who are the board of directors of JSFL ?"
# To skip query expansion uncomment the below line and comment the following 2 lines.
queries = [user_query]
# queries = generate_multiple_queries(user_query)
# print(queries)


########################################################################################################################

# 2. Query the vector store for the user query along with the generated queries and retrieve the top K chunks.
# 3. Rerank - Multiple options

# Note that the results from the pgvector are sorted on the basis of cosine distance which is inverse to cosine similarity.
# Hence, the lowest score has highest similarity and the highest score has lower similarity.
# If we have to convert this cosine distance to cosine similarity, we can use the following function:
# cosine_similarity = 1 - cosine_distance

### OPTION 1: Reciprocal Rank fusion
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def invoke_reciprocal_rank_fusion():
    vector_store = init_pg_vectorstore(recreate_collection=False)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    reciprocal_rank_fusion_chain = retriever.map() | reciprocal_rank_fusion
    reranked_results = reciprocal_rank_fusion_chain.invoke(queries)

    return reranked_results

def print_results_reciprocal_rank_fusion(reranked_results):
    for doc, score in reranked_results:
        print("*" * 20)
        print(f"Score: {score}")
        print(f"Page number: {doc.metadata['page_number']}")
        print(doc.page_content)

# reranked_results = invoke_reciprocal_rank_fusion()
# reranked_docs = [doc for doc, _ in reranked_results]
# print_results_reciprocal_rank_fusion(reranked_results)

### OPTION 2 : Using cross encoder LLM model - BAAI/bge-reranker-base
def rerank_docs_with_bge_reranker():
    vector_store = init_pg_vectorstore(recreate_collection=False)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compression_retriever_chain = compression_retriever.map()
    # For every generated query, we have reranked the results.
    compressed_docs_lists = compression_retriever_chain.invoke(queries)
    # Now, we will again rerank these results against the actual user query.
    # First. Flatten the compressed_docs_lists
    compressed_docs_initial_list = list(chain.from_iterable(compressed_docs_lists))
    # Using a dictionary to keep only unique documents based on id
    unique_documents = {compressed_doc_initial.id: compressed_doc_initial for compressed_doc_initial in
                        compressed_docs_initial_list}.values()
    unique_documents = list(unique_documents)
    compressed_docs = compressor.compress_documents(unique_documents, user_query)
    return compressed_docs


def print_bge_rerank_results(reranked_docs):
    for doc in reranked_docs:
        print("*" * 20)
        print(f"Page number: {doc.metadata['page_number']}")
        print(doc.page_content)


# reranked_docs = rerank_docs_with_bge_reranker()
# print_bge_rerank_results(reranked_docs)

### OPTION 3 : Using cross encoder LLM model - Jina Reranker
def rerank_docs_with_jina_reranker():
    vector_store = init_pg_vectorstore(recreate_collection=False)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    model = HuggingFaceCrossEncoder(model_name="jinaai/jina-reranker-v2-base-multilingual",
                                    model_kwargs={"trust_remote_code": True})
    compressor = CrossEncoderReranker(model=model, top_n=3)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compression_retriever_chain = compression_retriever.map()
    # For every generated query, we have reranked the results.
    compressed_docs_lists = compression_retriever_chain.invoke(queries)
    # Now, we will again rerank these results against the actual user query.
    # First. Flatten the compressed_docs_lists
    compressed_docs_initial_list = list(chain.from_iterable(compressed_docs_lists))
    # Using a dictionary to keep only unique documents based on id
    unique_documents = {compressed_doc_initial.id: compressed_doc_initial for compressed_doc_initial in
                        compressed_docs_initial_list}.values()
    unique_documents = list(unique_documents)
    compressed_docs = compressor.compress_documents(unique_documents, user_query)
    return compressed_docs


def print_jina_rerank_results(reranked_docs):
    for doc in reranked_docs:
        print("*" * 20)
        print(f"Page number: {doc.metadata['page_number']}")
        print(doc.page_content)


#reranked_docs = rerank_docs_with_jina_reranker()
# print_jina_rerank_results(reranked_docs)

### Option 4: Using EmbeddingsFilter Via Pipeline.
def rerank_docs_with_embeddings_filter(k=3):
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small", api_version="2024-02-01")
    vector_store = init_pg_vectorstore(recreate_collection=False)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    similarity_thresold = 0.38
    relevance_filter = EmbeddingsFilter(embeddings=embeddings,
                                        similarity_threshold=similarity_thresold)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )
    contextual_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )
    contextual_retriever_chain = contextual_retriever.map()
    # For every generated query, we have reranked the results.
    compressed_docs_lists = contextual_retriever_chain.invoke(queries)
    # Now, we will again rerank these results against the actual user query.
    # First. Flatten the compressed_docs_lists
    compressed_docs_initial_list = list(chain.from_iterable(compressed_docs_lists))
    # Using a dictionary to keep only unique documents based on id
    unique_documents = {compressed_doc_initial.metadata["id"]: compressed_doc_initial for compressed_doc_initial in
                        compressed_docs_initial_list}.values()
    unique_documents = list(unique_documents)
    compressed_docs = pipeline_compressor.compress_documents(unique_documents, user_query)
    if len(compressed_docs) == 0:
        return []
    # Return top K results
    if len(compressed_docs) < k:
        k = len(compressed_docs)
    return compressed_docs[:k]


def print_embeddings_filter_results(reranked_docs):
    for doc in reranked_docs:
        print("*" * 20)
        print(f"Page number: {doc.metadata['page_number']}")
        print(doc.page_content)


# reranked_docs = rerank_docs_with_embeddings_filter(k=5)
# print_embeddings_filter_results(reranked_docs)

### Option 5: Rerank Using LLM.
class Passage(BaseModel):
    """Passage Details"""
    passage_id: str = Field(description="Passage Metadata")
    passage_content: str = Field(description="Passage Content")


class Passages(BaseModel):
    """Reranked passages"""
    passages: List[Passage] = Field(description="Reranked passages")


prompt_template: str = """You are an AI language model assistant. Your task is to rerank passages related to a query based on their relevance. 
The most relevant passages should be identified with high accuracy. 
You should only pick at max {keep_top_k} passages. Just return the reranked passages content along with its id without any other text.

## Conditions:
1. If the passage contains irrelevant information to the query, then avoid it even if it is in the top {keep_top_k} passages.
2. If none of the passages contains relevant information to the query, then you are allowed to return empty results.

Question:
{question}

The following passages should be reranked based on the relevance to the provided question.
Passages: 
{passages}


"""


def rerank_based_on_llm():
    vector_store = init_pg_vectorstore(recreate_collection=False)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    retriever_chain = retriever.map()
    docs_lists = retriever_chain.invoke(queries)
    docs = list(chain.from_iterable(docs_lists))
    unique_documents = {doc.id: doc for doc in docs}.values()
    unique_documents = list(unique_documents)

    # Format documents to feed into LLM.
    formatted_documents = f"\n{'-' * 100}\n".join(
        [f"Passage {i + 1}:\n\n Passage Id: {d.id}\n Passage Content:\n" + d.page_content for i, d in enumerate(docs)]
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages = prompt.format_messages(keep_top_k=4, question=user_query, passages=formatted_documents)
    model_id = "gpt-4o"
    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)
    response = azure_openai_model.with_structured_output(Passages, method="json_schema", strict=True).invoke(messages)
    print("Number of passages identified: ", len(response.passages))
    reranked_passages_with_metadata = []
    for i, passage in enumerate(response.passages):
        passage_id = passage.passage_id
        passage_content = passage.passage_content
        # Extract metadata from unique_documents for the passage_id
        for doc in unique_documents:
            if doc.id == passage_id:
                passage_metadata = doc.metadata
                break
        reranked_passages_with_metadata.append((passage_metadata, passage_content))
    return reranked_passages_with_metadata


def print_reranked_passages_results(reranked_passages_with_metadata):
    for i, (passage_metadata, passage_content) in enumerate(reranked_passages_with_metadata):
        print(f"Passage {i + 1}:\n{passage_metadata}\n{passage_content}\n{'-' * 100}\n")


# reranked_passages_with_metadata = rerank_based_on_llm()
# reranked_docs = [Document(page_content=passage_content, metadata=passage_metadata)
#                  for i, (passage_metadata, passage_content) in enumerate(reranked_passages_with_metadata)]
# print_reranked_passages_results(reranked_passages_with_metadata)

### Option 6: Hybrid Retrieval (BM25 + Embeddings)

# We use ParadeDB that contains pgvector and pg_search and some advanced extensions.
# Refer to the fork version here - https://github.com/scchengaiah/paradedb-fork
# Docker command - https://github.com/scchengaiah/paradedb-fork/blob/dev/docker/standalone-docker-run.sh
# Docs - https://docs.paradedb.com/documentation/guides/hybrid

def hybrid_retrieval():
    ## We use hardcoded SQL query on langchain_pg_embedding table. For actual implementation, create customized implementation
    ## extending the langchain abstractions for standard usage.
    sql = """
    WITH semantic_search AS (
        SELECT id, RANK () OVER (ORDER BY embedding <=> %(embedding)s::vector) AS rank
        FROM langchain_pg_embedding ORDER BY embedding <=> %(embedding)s::vector LIMIT 20
    ),
    bm25_search AS (
        SELECT id, RANK () OVER (ORDER BY paradedb.score(id) DESC) as rank
        FROM langchain_pg_embedding WHERE document @@@ %(query)s LIMIT 20
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
    LIMIT 5;
    """

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small", api_version="2024-02-01")
    query_embeddings = embeddings.embed_query(user_query)
    # Use a smaller value of k when you want to give more weight to the keyword search (BM25) results.
    # Try setting k between 5 and 20. This keeps keyword relevance as the dominant factor but still allows semantic scores to contribute.

    # Use a larger value of k to give more weight to the semantic search results.
    # Set k between 60 and 100. Higher values make scores less sensitive to BM25 rank differences, letting the semantic search results have more influence.

    # A value of k between 30 and 60 is typically ideal for balancing BM25 and semantic search
    # Lower end (30-40): Slightly leans toward keyword relevance if that’s your preference.
    # Higher end (50-60): Slightly favors semantic similarity but retains substantial keyword influence.
    # Starting Value: k=50 is often a solid middle-ground starting point. This lets you achieve a reasonable balance, where both ranking lists contribute, yet neither dominates.
    k = 10
    db_host = "172.31.60.199"
    db_user = "admin"
    db_password = "admin"
    db_name = "postgres"
    db_port = "25432"
    conn = psycopg.connect(dbname=db_name, 
                            user=db_user, password=db_password, host=db_host, port=db_port,
                            autocommit=True)
    results = conn.execute(sql, {'query': user_query, 'embedding': query_embeddings, 'k': k}).fetchall()

    reranked_docs = []
    print(f"No of rows returned: {len(results)}")
    for row in results:
        reranked_docs.append(Document(page_content=row[2], metadata=row[4]))
    return reranked_docs

def print_hybrid_results(reranked_docs):
    for doc in reranked_docs:
        print("*" * 20)
        print(f"Page number: {doc.metadata['page_number']}")
        print(doc.page_content)

reranked_docs = hybrid_retrieval()
# print_hybrid_results(reranked_docs)

########################################################################################################################

############################################## Advanced Retrieval - END ################################################

############################################## Response Generation - START #############################################

########################################################################################################################
### Option 1:
# 1. Send the retrieved context to the LLM with appropriate prompt to answer the question.

def invoke_llm_rag_with_textual_context():
    prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don’t know the answer, just say that you cannot answer the question due to insufficient context without providing any additional information. "
    "Try to keep the answer concise and elaborate only if the question demands it.\n\n"
    "Question: {question}\n\n"
    "Context: \n{context}\n\n"
    "Answer:\n")

    formatted_context = "\n\n".join(
    [doc.page_content for i, doc in enumerate(reranked_docs)])

    messages = prompt.format_messages(question=user_query, context=formatted_context)
    print("*" * 20)
    print("LLM INPUT:")
    print(messages[0].content)
    print("*" * 20)
    model_id = "gpt-4o-mini"
    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)
    response = azure_openai_model.invoke(messages)
    print("*" * 20)
    print("Response:")
    print(response.content)

# invoke_llm_rag_with_textual_context()

### Option 2:

# The below method consumes is performance intensive but, is more robust.
# We pass the pdf pages as images to the LLM that provides more context for the LLM to generate the answer.

def invoke_llm_rag_with_image_context():

    rag_image_as_context_prompt = ("You are an assistant for question-answering tasks. Use the provided images as context to answer the question. "
    "If you don’t know the answer, just say that you cannot answer the question due to insufficient context without providing any additional information. "
    "Try to keep the answer concise and elaborate only if the question demands it.\n\n"
    "Question: {question}\n\n"
    "Answer:\n")

    unique_image_paths = set()

    for doc in reranked_docs:
        unique_image_paths.add(doc.metadata['image_path'])

    content = []
    for image_path in unique_image_paths:
        # Read the image file and encode it to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
            })

    content.append({"type": "text", "text": rag_image_as_context_prompt.format(question=user_query)})

    messages = HumanMessage(content)

    model_id = "gpt-4o-mini"
    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)
    response = azure_openai_model.invoke([messages])
    print("*" * 20)
    print("Response:")
    print(response.content)

# invoke_llm_rag_with_image_context()

########################################################################################################################


############################################## Response Generation - END #############################################
