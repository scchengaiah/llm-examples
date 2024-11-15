## The below implementation contains multiple code snippets that leverages the following aspects.
## 1. Convert the PDF to images.
## 2. Use LLM to describe images in structured markdown format.
## 3. Leverage Multi-vector retriever to ingest data to the vector store (Minimal Chunks) and doc store (Each page of the PDF as a parent doc).
## 4. Retrieve max number of chunks from vector store and consider the parent docs of those child chunks for LLM context.
## 5. Leverage ReRanker to fetch top k docs. (Cohere Reranker works best).
## 6. Invoke LLM with the Reranked docs to generate response.

## TODO

##   COMPLETED:
## - Use Cohere Reranker on Azure to reduce local dependency. - DONE (Refer to .env.example or .env file for the endpoint related info)

##   IN PROGRESS:
## - To incorporate custom PGRetrieval strategy to use BM25 + Semantic Search for better recall via ParadeDB.
## - The output of the BM25 + Semantic Search shall then be leveraged by the MultiVector Retriever.
##   (https://python.langchain.com/docs/integrations/stores/)
## - Setup a customized implementation to replace doc store with Postgresql specific implementation for better scalability.
## - Setup an end to end example leveraging all these aspects into a single application. (Preferably Streamlit to demonstrate the quality of the RAG implementation.)

import base64
import json
from math import ceil
import os
from typing import List
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters.markdown import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_text_splitters import SpacyTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
# https://github.com/PrithivirajDamodaran/FlashRank
# https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_cohere import CohereRerank
from langchain_together import ChatTogether
import psycopg
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import shutil
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers import ContextualCompressionRetriever
from itertools import chain
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = "./data"
image_dir = "./data/images"
temp_dir = "./temp"
chunk_dir= os.path.join(temp_dir, "chunks")
pdf_file = os.path.join(data_dir, "jio-financial-services-annual-report-2023-2024-small.pdf")
# pdf_file = "C:/Users/20092/Downloads/test2/20230725_Designrules_PORSCHE_V01.pdf"

headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4")
    ]

def pdf_to_markdown ():
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(chunk_dir, exist_ok=True)
    else:
        shutil.rmtree(image_dir)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(chunk_dir, exist_ok=True)

    def convert_pdf_to_images(pages = None):
        import fitz

        os.makedirs(image_dir, exist_ok=True)

        doc = fitz.open(pdf_file)

        if not pages:
            pages = list(range(len(doc)))

        for i, page_number in enumerate(pages):  # Iterate through the pages
            page = doc.load_page(page_number)  # Load the page
            pix = page.get_pixmap(dpi=150)  # Render the page to a pixmap
            pix.save(os.path.join(image_dir, f"page-{i + 1}.png"))  # Save the pixmap as a PNG file

        doc.close()

    # Index starts from zero, If the page number in the pdf is 1, then it should be specified as 0 here.
    pages = [0,1,2,18,19,20]
    pages = []
    convert_pdf_to_images(pages)
    image_path_list = os.listdir(image_dir)

    openai_model_id = "gpt-4o-mini"
    together_model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"

    azure_openai_model = AzureChatOpenAI(model=openai_model_id, max_tokens=8192)
    together_llama_90b_vision_model = ChatTogether(model = together_model_id, temperature=0, max_tokens=2048)



    def multi_context_llm_invocation():

        class TextualContent(BaseModel):
            title: str = Field(description="Relevant title that supports the textual representation.")
            textual_representation: str = Field(description="Textual representation of the image in markdown format.")
            image_path: str = Field(description = "Image that was primarily considered to generate the textual representation")
            able_to_follow_instruction: bool = Field(description= "Confirmation to assert that the instructions are followed as provided.")
            additional_information: str = Field(description= "Additional information only if the instructions are deviated for some reason while generating the textual representation")

        first_page_prompt = """To preserve the cohesiveness between multiple pages of the document, you shall be provided with two images. The second image should only be considered as the context to determine the cohesivess of the information flow for the first image. The first image is the one that you should focus upon to generate textual based content. If the textual content for the first image lacks context by its own and if the second image can add value or maintain cohesiveness then consider it for generating the content. For example: If you have image1, image2 then image1 should be your point of focus, the image2 act as an supporting context for image1.
        """

        last_page_prompt = """To preserve the cohesiveness between multiple pages of the document, you shall be provided with two images. The first image should only be considered as the context to determine the cohesivess of the information flow for the second image. The second image is the one that you should focus upon to generate textual based content. If the textual content for the second image lacks context by its own and if the first image can add value or maintain cohesiveness then consider it for generating the content. For example: If you have image1, image2 then image2 should be your point of focus, the image1 act as an supporting context for image2.
        """

        middle_page_prompt = """To preserve the cohesiveness between multiple pages of the document, you shall be provided with three images. The first image and the last image should only be considered as the context to determine the cohesivess of the information flow for the middle image. The middle image is the one that you should focus upon to generate textual based content. If the textual content for the middle image lacks context by its own and if the first and last image can add value or maintain cohesiveness then consider it for generating the content. For example: If you have image1, image2 and image3, then image2 should be your point of focus, the image1 and image3 act as an supporting context for image2."""

        image_to_text_prompt = """You are an expert in analyzing images and convert it into a textual representation. The images are created by converting the documents into an image representation. This means, every image provided to you should be seen as a page from the document. This page that is represented as an image may contain text, images, graphs, complex tables and other form of information representation. 

        Since these images are created from the document, all document related aspects should be taken into consideration when analyzing these images. For example: It may contain company logo, Table of Contents, References, Text represented in a multi-column layout, Links, Headings, Sections, and paragraphs.

        You have to consider these document related aspects mentioned above and convert the image into a detailed textual representation. You should try to format your response in a markdown based format at the same time preserving the sequence of the content. Act as a text extractor whenever there is a textual content present in the image so that explicit generation is not required. For certain sections of the document that contains complex information representations such as tables and images, you have to convert into a readable textual representation. For example, Tables into markdown tables and for images generate detailed textual description.

        {dynamic_prompt_for_multiple_images}

        Below are the images provided in the same sequence that is provided to you:
        {uploaded_image_paths}

        Your response should be supported by the appropriate title for the content you are generating so that the entire content reflects the title.

        For some reason, if you could not able to perform based on the provided instructions or if you have some ambiguity while generating the content, mention the same in the additional_information attribute, set this to empty if no instructions are deviated. Also, update the able_to_follow_instruction boolean attribute to true so that I can be confident that the content generated is acceptable. You should also update the image_path attribute with the image that you have primarily considered to generate the content."""
        
        response_list = []

        for i, _ in enumerate(image_path_list):
            # Increment by one to start from 1
            idx = i+1
            
            if idx == 1 and len(image_path_list) == 1:
                print("This is the only page of the document")
                image_file_1 = f"page-{idx}.png"
                image_path_1 = os.path.join(image_dir, image_file_1)
                print(f"Load {image_file_1}")

                # Read the image file and encode it to base64
                with open(image_path_1, "rb") as image_file:
                    encoded_string_1 = base64.b64encode(image_file.read()).decode('utf-8')

                uploaded_image_paths = "\n".join([image_file_1])
                

                messages = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_1}"},
                        },
                        {"type": "text", "text": image_to_text_prompt.format(dynamic_prompt_for_multiple_images="",
                            uploaded_image_paths=uploaded_image_paths)},
                    ],
                )


            elif idx == 1 and len(image_path_list) > 1:
                print("This is the first page of the document")
                image_file_1 = f"page-{idx}.png"
                image_path_1 = os.path.join(image_dir, image_file_1)
                image_file_2 = f"page-{idx+1}.png"
                image_path_2 = os.path.join(image_dir, image_file_2)

                print(f"Load {image_file_1} and {image_file_2}")

                # Read the image file and encode it to base64
                with open(image_path_1, "rb") as image_file:
                    encoded_string_1 = base64.b64encode(image_file.read()).decode('utf-8')
                # Read the image file and encode it to base64
                with open(image_path_2, "rb") as image_file:
                    encoded_string_2 = base64.b64encode(image_file.read()).decode('utf-8')

                uploaded_image_paths = "\n".join([image_file_1, image_file_2])

                messages = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_1}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_2}"},
                        },
                        {"type": "text", "text": image_to_text_prompt.format(dynamic_prompt_for_multiple_images=first_page_prompt,
                            uploaded_image_paths=uploaded_image_paths)},
                    ],
                )

            elif idx == len(image_path_list):
                print("This is the last page of the document")
                image_file_1 = f"page-{idx-1}.png"
                image_path_1 = os.path.join(image_dir, image_file_1)
                image_file_2 = f"page-{idx}.png"
                image_path_2 = os.path.join(image_dir, image_file_2)
                print(f"Load {image_file_1} and {image_file_2}")

                # Read the image file and encode it to base64
                with open(image_path_1, "rb") as image_file:
                    encoded_string_1 = base64.b64encode(image_file.read()).decode('utf-8')
                # Read the image file and encode it to base64
                with open(image_path_2, "rb") as image_file:
                    encoded_string_2 = base64.b64encode(image_file.read()).decode('utf-8')

                uploaded_image_paths = "\n".join([image_file_1, image_file_2])

                messages = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_1}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_2}"},
                        },
                        {"type": "text", "text": image_to_text_prompt.format(dynamic_prompt_for_multiple_images=last_page_prompt,
                            uploaded_image_paths=uploaded_image_paths)},
                    ],
                )
            else:
                print("This is a middle page of the document")
                image_file_1 = f"page-{idx-1}.png"
                image_path_1 = os.path.join(image_dir, image_file_1)
                image_file_2 = f"page-{idx}.png"
                image_path_2 = os.path.join(image_dir, image_file_2)
                image_file_3 = f"page-{idx+1}.png"
                image_path_3 = os.path.join(image_dir, image_file_3)
                print(f"Load {image_file_1}, {image_file_2} and {image_file_3}")


                # Read the image file and encode it to base64
                with open(image_path_1, "rb") as image_file:
                    encoded_string_1 = base64.b64encode(image_file.read()).decode('utf-8')
                # Read the image file and encode it to base64
                with open(image_path_2, "rb") as image_file:
                    encoded_string_2 = base64.b64encode(image_file.read()).decode('utf-8')
                # Read the image file and encode it to base64
                with open(image_path_3, "rb") as image_file:
                    encoded_string_3 = base64.b64encode(image_file.read()).decode('utf-8')

                uploaded_image_paths = "\n".join([image_file_1, image_file_2, image_file_3])

                messages = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_1}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_2}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string_3}"},
                        },
                        {"type": "text", "text": image_to_text_prompt.format(dynamic_prompt_for_multiple_images=middle_page_prompt,
                            uploaded_image_paths=uploaded_image_paths)},
                    ],
                )
            print("Sending messages to LLM...")
            response = azure_openai_model.with_structured_output(TextualContent).invoke([messages])
            response_list.append(response)

        print("*" * 50)
        for response in response_list:
            print("*" * 20)
            print("Image_path:", response.image_path)
            print("Able to follow instruction:", response.able_to_follow_instruction)
            print("Additional information:", response.additional_information)
            print("CONTENT:")
            print(response.textual_representation)


    def normal_llm_invocation(open_ai = True):
        class TextualContent(BaseModel):
            title: str = Field(description="Relevant title that supports the textual representation.")
            textual_representation: str = Field(description="Textual representation of the image in plain text format.")
            image_path: str = Field(description = "Image that was primarily considered to generate the textual representation")
            able_to_follow_instruction: bool = Field(description= "Confirmation to assert that the instructions are followed as provided.")
            additional_information: str = Field(description= "Additional information only if the instructions are deviated for some reason while generating the textual representation")

        image_to_text_prompt_openai = """You are an expert in analyzing images and convert it into a textual representation. The images are created by converting the documents into an image representation. This means, every image provided to you should be seen as a page from the document. This page that is represented as an image may contain text, images, graphs, complex tables and other form of information representation. 

        Since these images are created from the document, all document related aspects should be taken into consideration when analyzing these images. For example: It may contain company logo, Table of Contents, References, Text represented in a multi-column layout, Links, Headings, Sections, and paragraphs. If there are any other information that does not add significant value to the content, you can ignore them. For example: Page numbers.

        You have to consider these document related aspects mentioned above and convert the image into a detailed textual representation. The textual representation should be in the same language as the document. You should try to format your response in a markdown format by adding appropriate headings or sections and at the same time preserving the sequence of the content. Always begin your heading from second level for the generated content eventhough the actual content is in the first level. For example: ## Section 1.
        Maintain cohesiveness for the generated content without scattering similar information across multiple sections and always try to keep the sections related to the same topic together.
        
        Act as a text extractor, whenever there is a textual content present in the image, use it as is so that explicit generation is not required. For certain sections of the document that contains complex information representations such as tables and images, you have to convert into a readable textual representation. For example, Tables into markdown table structure and for images, generate detailed textual description.

        Image path in the local file system:
        {uploaded_image_path}

        Your response should be supported by the appropriate title for the content you are generating so that the title reflects the entire content.

        For some reason, if you could not able to perform based on the provided instructions or if you have some ambiguity while generating the content, mention the same in the additional_information attribute, set this to empty if no instructions are deviated. Also, update the able_to_follow_instruction boolean attribute to true so that I can be confident that the content generated is acceptable. You should also update the image_path attribute with the image that you have primarily considered to generate the content."""

        ## PROMPT not able to output in JSON.
        image_to_text_prompt_llama = """You are an expert in analyzing images and convert it into a textual representation. The images are created by converting the documents into an image representation. This means, every image provided to you should be seen as a page from the document. This page that is represented as an image may contain text, images, graphs, complex tables and other form of information representation. 

        Since these images are created from the document, all document related aspects should be taken into consideration when analyzing these images. For example: It may contain company logo, Table of Contents, References, Text represented in a multi-column layout, Links, Headings, Sections, and paragraphs. If there are any other information that does not add significant value to the content, you can ignore them. For example: Page numbers.

        You have to consider these document related aspects mentioned above and convert the image into a detailed textual representation. You should try to format your response in a markdown format by adding appropriate headings or sections and at the same time preserving the sequence of the content. Always begin your heading from second level for the generated content eventhough the actual content is in the first level. For example: ## Section 1. Maintain cohesiveness for the generated content without scattering similar information across multiple sections and always try to keep the sections related to the same topic together.
        
        Act as a text extractor, whenever there is a textual content present in the image, use it as is so that explicit generation is not required. For certain sections of the document that contains complex information representations such as tables and images, you have to convert into a readable textual representation. For example, Tables into markdown table structure and for images, generate detailed textual description.

        Image path in the local file system:
        {uploaded_image_path}

        Your response should be supported by the appropriate title for the content you are generating so that the title reflects the entire content.

        For some reason, if you could not able to perform based on the provided instructions or if you have some ambiguity while generating the content, mention the same in the additional_information attribute, set this to empty if no instructions are deviated. Also, update the able_to_follow_instruction boolean attribute to true so that I can be confident that the content generated is acceptable. You should also update the image_path attribute with the image that you have primarily considered to generate the content.
        
        You should consolidate all the content based on the above instructions and generate the final response in json format.

        ## EXAMPLE RESPONSE:
        {json_format}

        ## JSON RESPONSE:
        ```json

        """

        json_format = {
            "title": "Relevant title that supports the textual representation.",
            "textual_representation": "Textual representation of the image in markdown format.",
            "image_path": "Image that was primarily considered to generate the textual representation",
            "able_to_follow_instruction": "Confirmation to assert that the instructions are followed as provided.",
            "additional_information": "Additional information only if the instructions are deviated for some reason while generating the textual representation."
        }


        batch_messages = []
        for i, _ in enumerate(image_path_list):
            # Increment by one to start from 1
            idx = i+1

            image_file_name = f"page-{idx}.png"
            image_file_path = os.path.join(image_dir, image_file_name)

            # Read the image file and encode it to base64
            with open(image_file_path, "rb") as image_file:
                encoded_String = base64.b64encode(image_file.read()).decode('utf-8')        

            if open_ai:
                messages = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_String}"},
                        },
                        {"type": "text", "text": image_to_text_prompt_openai.format(uploaded_image_path=image_file_path)},
                    ],
                )
            else:
                messages = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_String}"},
                        },
                        {"type": "text", "text": image_to_text_prompt_llama.format(uploaded_image_path=image_file_path, 
                                                                                json_format=json.dumps(json_format))},
                    ],
                )
            print("Processing page:", image_file_path)
            batch_messages.append([messages])
        
        if open_ai:
            batch_response = azure_openai_model.with_structured_output(TextualContent).batch(batch_messages,
                                                                                                config={"max_concurrency": 5})
        else:
            batch_response = together_llama_90b_vision_model.batch(batch_messages, config={"max_concurrency": 5})  
            
        ### if not open_ai:
        ###     for response in batch_response:
        ###         print(response)
        ### else:
        ###     for response in batch_response:
        ###             print("*" * 20)
        ###             print("Image_path:", response.image_path)
        ###             print("Able to follow instruction:", response.able_to_follow_instruction)
        ###             print("Additional information:", response.additional_information)
        ###             print("TITLE: ", response.title)
        ###             print("CONTENT:")
        ###             print(response.textual_representation)
        
        return batch_response

    def response_to_markdown(batch_response):
        for response in batch_response:
            markdown_content = ""
            image_path = response.image_path
            page_number = os.path.basename(image_path).split('-')[1].split('.')[0]
            markdown_path = os.path.join(chunk_dir, f"page-chunk-output-{page_number}.md")
            markdown_content += f"# {response.title}\n\n"
            markdown_content += response.textual_representation + "\n\n"
            markdown_content += "---\n\n" # Page break
            with open(markdown_path, "wb") as f:
                f.write(markdown_content.encode("utf-8"))

    batch_response = normal_llm_invocation(open_ai=True)
    response_to_markdown(batch_response)


pdf_to_markdown()

def reconstruct_markdown(splits, headers_to_split_on):
    # Dictionary to store the structured content
    document_structure = {}
    headers_to_split_on_prefixes = [header[0] + ' ' for header in headers_to_split_on]
    for doc in splits:
        metadata = doc.metadata

        # Remove any markdown headers from the content
        # content = '\n'.join([line for line in doc.page_content.split('\n') 
        #                    if not line.strip().startswith('#')]).strip()
        # Remove only the specified markdown headers from the content
        content = '\n'.join([line for line in doc.page_content.split('\n') 
                            if not any(line.strip().startswith(prefix) for prefix in headers_to_split_on_prefixes)]).strip()
        if not content:  # Skip if there's no content after removing headers
            continue
            
        # Determine the deepest header level present in metadata
        header_levels = [k for k in metadata.keys() if k.startswith('header_')]
        deepest_header = max(header_levels, key=lambda x: int(x.split('_')[1])) if header_levels else None
        
        if not deepest_header:
            continue
            
        # Get all header values up to the deepest level
        current_path = []
        for i in range(1, int(deepest_header.split('_')[1]) + 1):
            header_key = f'header_{i}'
            if header_key in metadata:
                current_path.append(metadata[header_key])
        
        # Navigate/create nested dictionary structure
        current_dict = document_structure
        for i, header in enumerate(current_path[:-1]):
            if header not in current_dict:
                current_dict[header] = {'content': '', 'subsections': {}}
            current_dict = current_dict[header]['subsections']
            
        # Add content to the deepest level
        if current_path:
            last_header = current_path[-1]
            if last_header not in current_dict:
                current_dict[last_header] = {'content': '', 'subsections': {}}
            if content:  # Only add non-empty content
                current_dict[last_header]['content'] += f"{content}\n"

    # Function to generate markdown from the structure
    def generate_markdown(structure, level=1):
        markdown = ""
        for header, data in structure.items():
            markdown += f"{'#' * level} {header}\n"
            if data['content']:
                markdown += f"{data['content'].strip()}\n\n"
            if data['subsections']:
                markdown += generate_markdown(data['subsections'], level + 1)
        return markdown

    # Generate the final markdown
    return generate_markdown(document_structure)

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

def ingest_to_vectorstore(recreate_collection=False):
    final_docs = []
    for i, _ in enumerate(chunk_dir):
        # Increment by one to start from 1
        idx = i+1
        filename = f"page-chunk-output-{idx}.md"
        file_path = os.path.join(chunk_dir, filename)
        if os.path.exists(file_path):
            with open(os.path.join(chunk_dir, file_path), 'rb') as file:
                print("Processing file:", file_path)
                markdown_text = file.read().decode('utf-8')

            # The entire markdown_text should be part of the parent chunk.
            # Split the markdown_text into chunks of 1000 characters that should act as child chunks.
            # During retrieval, the parent chunk should be identified based on the child chunk and shall be passed as context to the LLM.


            # MD splits
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, return_each_line=False, strip_headers=True
            )
            md_header_splits = markdown_splitter.split_text(markdown_text)

            ### for i, text_split in enumerate(md_header_splits):
            ###     print("-" * 80)
            ###     print(f"Split {i}:\n\n {text_split}\n\n")

            ### print("-" * 80)
            ### print("Markdown Header Splits - RECURSIVE")
            ### print("-" * 80)

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
                doc.metadata["type"] = "MARKDOWN"
                doc.metadata["file_name"] = os.path.basename(pdf_file)
                doc.metadata["file_path"] = pdf_file
                doc.metadata["page_number"] = idx
                doc.metadata["image_path"] = os.path.join(image_dir, f"page-{idx}.png")
                modified_doc = Document(page_content=new_content, metadata=doc.metadata)
                modified_splits.append(modified_doc)

                ### for i, doc in enumerate(modified_splits):
                ###     print("-" * 80)
                ###     print(f"Split {i}:\n\n Metadata: \n{doc.metadata}\n\n Content: \n{doc.page_content}\n\n")
            final_docs.extend(modified_splits)
    


    with open(os.path.join(temp_dir, "chunk-output.txt"), "w", encoding='utf-8') as outfile:
        for i, doc in enumerate(final_docs):
            content = "-" * 80
            content += f"\n\nSplit {i}:\n\nMetadata:\n{doc.metadata}\n\nContent:\n{doc.page_content}\n\n"
            outfile.write(content)
        
    vectorstore = init_pg_vectorstore(recreate_collection)
    print(f"Number of documents to ingest into the vector store: {len(final_docs)}")
    vectorstore.add_documents(final_docs, ids=[doc.metadata['id'] for doc in final_docs])
    print("Documents ingested into the vector store successfully.")

# https://python.langchain.com/docs/how_to/multi_vector/
def ingest_to_vectorstore_with_multi_vector_retriever(recreate_collection=True):
    # Each item corresponds to a page of markdown content.
    parent_docs = []
    # Final docs corresponds to the total chunks for all the pages.
    final_docs = []
    # The storage layer for the parent documents
    # Other possibilities - https://python.langchain.com/docs/integrations/stores/
    byte_store = LocalFileStore(os.path.join(temp_dir, "byte-store"))
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore = init_pg_vectorstore(recreate_collection),
        byte_store=byte_store,
        id_key=id_key,
    )

    for i, _ in enumerate(os.listdir(chunk_dir)):
        # Increment by one to start from 1
        idx = i+1
        filename = f"page-chunk-output-{idx}.md"
        file_path = os.path.join(chunk_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                print("Processing file:", file_path)
                markdown_text = file.read().decode('utf-8')

            # The entire markdown_text should be part of the parent chunk.
            # Split the markdown_text into chunks of 1000 characters that should act as child chunks.
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
            parent_docs.append(Document(page_content=markdown_text, metadata=parent_doc_metadata))

            # MD splits
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, return_each_line=False, strip_headers=True
            )
            md_header_splits = markdown_splitter.split_text(markdown_text)

            ### for i, text_split in enumerate(md_header_splits):
            ###     print("-" * 80)
            ###     print(f"Split {i}:\n\n {text_split}\n\n")

            ### print("-" * 80)
            ### print("Markdown Header Splits - RECURSIVE")
            ### print("-" * 80)

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

                ### for i, doc in enumerate(modified_splits):
                ###     print("-" * 80)
                ###     print(f"Split {i}:\n\n Metadata: \n{doc.metadata}\n\n Content: \n{doc.page_content}\n\n")
            final_docs.extend(modified_splits)
    


    ### with open(os.path.join(temp_dir, "chunk-output.txt"), "w", encoding='utf-8') as outfile:
    ###     for i, doc in enumerate(final_docs):
    ###         content = "-" * 80
    ###         content += f"\n\nSplit {i}:\n\nMetadata:\n{doc.metadata}\n\nContent:\n{doc.page_content}\n\n"
    ###         outfile.write(content)
    
    # Ingest all the chunks into the vector store
    print(f"Number of documents to ingest into the vector store: {len(final_docs)}")
    retriever.vectorstore.add_documents(final_docs, ids=[doc.metadata['id'] for doc in final_docs])
    

    # Ingest all the chunks into the byte store
    doc_ids = [doc.metadata['id'] for doc in parent_docs]
    # Access the doc store through retriever since it takes care of serialization and de-serialization.
    retriever.docstore.mset(list(zip(doc_ids, parent_docs)))
    print("Documents ingested into the vector store and doc store successfully.")

def hybrid_retrieval(user_query):
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
    k = 60
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

def rerank_docs_with_embeddings_filter(user_query, k=20):
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small", api_version="2024-02-01")
    vector_store = init_pg_vectorstore(recreate_collection=False)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 50})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    similarity_thresold = 0.45
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
    compressed_docs_lists = contextual_retriever_chain.invoke([user_query])
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

user_query = "who are the board of directors of JSFL ? Explain about them."
# user_query = "what is the message from the management to investors ? Explain in detail who said what ?"
# user_query = "what are the services offered by JFSL ?"
# user_query = "Explain about JIO Financial app features."
# user_query = "Zulässige Kombination für Querschnittsprünge"

# ingest_to_vectorstore(recreate_collection=True)
# reranked_docs = hybrid_retrieval(user_query)
# reranked_docs = rerank_docs_with_embeddings_filter(user_query)

# ingest_to_vectorstore_with_multi_vector_retriever(recreate_collection=True)
multi_vector_retriever = MultiVectorRetriever(
        vectorstore=init_pg_vectorstore(recreate_collection=False),
        byte_store = LocalFileStore(os.path.join(temp_dir, "byte-store")),
        id_key="doc_id",
        search_type="similarity",
        # For the overall retriever, gets reflected for multi_vector_retriever.invoke(query) method.
        # score_threshold is only valid if search_type="similarity_score_threshold"
        search_kwargs= {"k": 25, "score_threshold": 0.5}
)
# The vector store alone from the multi_vector_retriever shall return smaller chunks fetched from the vector store.
# docs = multi_vector_retriever.vectorstore.similarity_search(user_query, k=20)
docs = multi_vector_retriever.invoke(user_query)

# Rank 2 :)
# model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base", model_kwargs={"trust_remote_code": True}) # Context length: 512

### compressor = CrossEncoderReranker(model=model, top_n=4)
# Less Performant - Rank 3 :)
### compressor = FlashrankRerank(top_n=4)
# Better performing ReRanking Model - Rank 1 :)
print("Using CohereRerank - START")
rerank_start_time = time.time()
compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=4)
docs = compressor.compress_documents(docs, user_query)
print("Using CohereRerank - END")
print(f"Time taken to rerank documents: {time.time() - rerank_start_time} seconds")

### for i, doc in enumerate(docs):
###     print("-" * 80)
###     print(f"\n\nSplit {i}:\n\nMetadata:\n{doc.metadata}\n\nContent:\n{doc.page_content.strip()}\n\n")

# POST RETRIEVAL DATA PROCESSING.
### print("-" * 80)
### print("CONSTRUCTED MARKDOWN")
# Only type of value markdown present in metadata should be passed to this function.
# Since, this function shall reconstruct the complete content based on the header values.
# All metadata values with other type shall be normally handled.
# final_markdown = reconstruct_markdown(reranked_docs, headers_to_split_on=headers_to_split_on)

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

# invoke_llm_rag_with_textual_context()

