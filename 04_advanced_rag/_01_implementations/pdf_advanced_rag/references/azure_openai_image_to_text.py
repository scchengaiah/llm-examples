# azure_openai_image_to_text.py
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import os
import base64

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
image_dir = os.path.join(data_dir, "images")
temp_dir = os.path.join(project_root, "temp")

image_path = os.path.join(image_dir, "page-20.png")

azure_model_id = "gpt-4o-mini"
azure_openai_vision_model = AzureChatOpenAI(model=azure_model_id, temperature=0, max_tokens=8192)

prompt = """As an expert in image analysis and textual conversion, your task is to analyze images representing pages from documents and convert them into detailed textual representations. The images may contain text, images, graphs, complex tables, and other forms of information representation. It is essential to consider all document-related aspects when analyzing these images, such as company logos, Table of Contents, references, text in multi-column layout, links, headings, sections, and paragraphs. Any other information that does not add significant value to the content can be ignored, such as page numbers and layout.

Your textual representation should be formatted in markdown format, with appropriate headings or sections to preserve the sequence of the content. Always begin your heading from the second level for the generated content, even if the actual content is in the first level. Maintain cohesiveness for the generated content without scattering similar information across multiple sections, and keep sections related to the same topic together.

When there is textual content present in the image, use it as is without explicit generation. For sections containing complex information representations like tables and images, convert them into readable textual representations. For example, convert tables into markdown table structure and provide detailed textual descriptions for images.

Your response should be supported by an appropriate title that reflects the overall content of the image. Start the title with a first-level header. For example: # Title."""

# Read the image file and encode it in base64 format
with open(image_path, "rb") as image_file:
    encoded_String = base64.b64encode(image_file.read()).decode('utf-8')

messages = HumanMessage(
    content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_String}"},
        },
        {"type": "text", "text": prompt},
    ],
)

content = ""
for chunk in azure_openai_vision_model.stream([messages]):
    content += chunk.content
    print(chunk.content, end="", flush=True)

with open(os.path.join(temp_dir, "image_output.md"), "wb") as f:
    f.write(content.encode('utf-8'))