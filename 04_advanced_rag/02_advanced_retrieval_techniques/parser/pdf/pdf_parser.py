from parser.pdf.helpers.pymupdf_rag import to_markdown
from llm.azure_openai_api import fetch_image_description_from_azure_openai
import pymupdf

class PDFParser:

    def __init__(self, file_obj, describe_images=False):
        self.file_obj = file_obj
        self.describe_images = describe_images

    def parse(self):
        doc = pymupdf.open(filetype="pdf", stream=self.file_obj.read())
        if self.describe_images:
            return to_markdown(doc, generate_img_desc=True,
                               generate_img_desc_llm_callable=fetch_image_description_from_azure_openai)
        else:
            return to_markdown(doc)

    async def parse_async(self):
        doc = pymupdf.open(filetype="pdf", stream=await self.file_obj.read())
        if self.describe_images:
            return to_markdown(doc, generate_img_desc=True,
                               generate_img_desc_llm_callable=fetch_image_description_from_azure_openai)
        else:
            return to_markdown(doc)
