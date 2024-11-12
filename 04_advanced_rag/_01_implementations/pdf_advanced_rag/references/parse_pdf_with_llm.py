import base64
import os
from typing import List
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import AzureChatOpenAI
from pydantic import BaseModel, Field


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
image_dir = os.path.join(data_dir, "images")
temp_dir = os.path.join(project_root, "temp")

if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)

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
pages = [18,19,20]
convert_pdf_to_images(pages)

image_path_list = os.listdir(image_dir)

model_id = "gpt-4o-mini"

azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)




response_list = []

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


def normal_llm_invocation():

    class TextualContent(BaseModel):
        title: str = Field(description="Relevant title that supports the textual representation.")
        textual_representation: str = Field(description="Textual representation of the image in plain text format.")
        image_path: str = Field(description = "Image that was primarily considered to generate the textual representation")
        able_to_follow_instruction: bool = Field(description= "Confirmation to assert that the instructions are followed as provided.")
        additional_information: str = Field(description= "Additional information only if the instructions are deviated for some reason while generating the textual representation")

    image_to_text_prompt = """You are an expert in analyzing images and convert it into a textual representation. The images are created by converting the documents into an image representation. This means, every image provided to you should be seen as a page from the document. This page that is represented as an image may contain text, images, graphs, complex tables and other form of information representation. 

    Since these images are created from the document, all document related aspects should be taken into consideration when analyzing these images. For example: It may contain company logo, Table of Contents, References, Text represented in a multi-column layout, Links, Headings, Sections, and paragraphs.

    You have to consider these document related aspects mentioned above and convert the image into a detailed textual representation. You should try to format your response in a plain text format by adding appropriate headings or sections and at the same time preserving the sequence of the content. Act as a text extractor, whenever there is a textual content present in the image, use it as is so that explicit generation is not required. For certain sections of the document that contains complex information representations such as tables and images, you have to convert into a readable textual representation. For example, Tables into markdown table structure and for images, generate detailed textual description.

    Image path in the local file system:
    {uploaded_image_path}

    Your response should be supported by the appropriate title for the content you are generating so that the title reflects the entire content.

    For some reason, if you could not able to perform based on the provided instructions or if you have some ambiguity while generating the content, mention the same in the additional_information attribute, set this to empty if no instructions are deviated. Also, update the able_to_follow_instruction boolean attribute to true so that I can be confident that the content generated is acceptable. You should also update the image_path attribute with the image that you have primarily considered to generate the content."""

    for i, _ in enumerate(image_path_list):
        # Increment by one to start from 1
        idx = i+1

        image_file_name = f"page-{idx}.png"
        image_file_path = os.path.join(image_dir, image_file_name)

        # Read the image file and encode it to base64
        with open(image_file_path, "rb") as image_file:
            encoded_String = base64.b64encode(image_file.read()).decode('utf-8')        

        messages = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_String}"},
                },
                {"type": "text", "text": image_to_text_prompt.format(uploaded_image_path=image_file_path)},
            ],
        )
        print("Processing page:", image_file_path)
        response = azure_openai_model.with_structured_output(TextualContent).invoke([messages])
        response_list.append(response)

    for response in response_list:
            print("*" * 20)
            print("Image_path:", response.image_path)
            print("Able to follow instruction:", response.able_to_follow_instruction)
            print("Additional information:", response.additional_information)
            print("TITLE: ", response.title)
            print("CONTENT:")
            print(response.textual_representation)

normal_llm_invocation()