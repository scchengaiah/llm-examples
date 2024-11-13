import base64
import json
import os
from typing import List
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_text_splitters.markdown import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_text_splitters import SpacyTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_together import ChatTogether
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import shutil


def pdf_to_markdown ():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    image_dir = os.path.join(data_dir, "images")
    temp_dir = os.path.join(project_root, "temp")
    chunk_dir= os.path.join(temp_dir, "chunks")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(chunk_dir, exist_ok=True)
    else:
        shutil.rmtree(image_dir)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(chunk_dir, exist_ok=True)

    pdf_file = os.path.join(data_dir, "jio-financial-services-annual-report-2023-2024-small.pdf")

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
    pages = [1,18,19,20]
    convert_pdf_to_images()

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

        You have to consider these document related aspects mentioned above and convert the image into a detailed textual representation. You should try to format your response in a markdown format by adding appropriate headings or sections and at the same time preserving the sequence of the content. Always begin your heading from second level for the generated content eventhough the actual content is in the first level. For example: ## Section 1. Limit your headings to third level even if the actual content is in the fourth level. For example: ### Section 3.
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
            
        if not open_ai:
            for response in batch_response:
                print(response)
        else:
            for response in batch_response:
                    print("*" * 20)
                    print("Image_path:", response.image_path)
                    print("Able to follow instruction:", response.able_to_follow_instruction)
                    print("Additional information:", response.additional_information)
                    print("TITLE: ", response.title)
                    print("CONTENT:")
                    print(response.textual_representation)
        
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


# pdf_to_markdown()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
temp_dir = os.path.join(project_root, "temp")
chunk_dir= os.path.join(temp_dir, "chunks")



azure_openai_embeddings = AzureOpenAIEmbeddings(model = "text-embedding-3-small", api_version="2023-05-15")

with open(os.path.join(chunk_dir, "page-chunk-output-20.md"), 'rb') as file:
    markdown_text = file.read().decode('utf-8')

headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4")
]
# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, return_each_line=False, strip_headers=True
)
md_header_splits = markdown_splitter.split_text(markdown_text)

for i, text_split in enumerate(md_header_splits):
    print("-" * 80)
    print(f"Split {i}:\n\n {text_split}\n\n")

print("-" * 80)
print("Markdown Header Splits - RECURSIVE")
print("-" * 80)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0
)

splits = text_splitter.split_documents(md_header_splits)

# First, create a new list to store the modified documents
modified_splits = []

# Iterate through the splits and modify each document to include headers as part of the content.
# This modified_splits should be considered for chunking purposes as it provides more context to the content based on the headings.
for doc in splits:
    # Create the header string by combining all headers in order
    header_content = ""
    for markdown_symbol, metadata_key in headers_to_split_on:
        if metadata_key in doc.metadata:
            header_content += f"{markdown_symbol} {doc.metadata[metadata_key]}\n"
    
    # Create a new document with headers prepended to the content
    new_content = header_content + doc.page_content
    
    # Create a new document with the same metadata but modified content
    modified_doc = Document(page_content=new_content, metadata=doc.metadata)
    modified_splits.append(modified_doc)

for i, doc in enumerate(modified_splits):
    print("-" * 80)
    print(f"Split {i}:\n\n Metadata: \n{doc.metadata}\n\n Content: \n{doc.page_content}\n\n")





print("-" * 80)
print("CONSTRUCTED MARKDOWN")
from rough import reconstruct_markdown
final_markdown = reconstruct_markdown(splits, headers_to_split_on=headers_to_split_on)

print(final_markdown)