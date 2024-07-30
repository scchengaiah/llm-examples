import os
from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.oxml.shape import CT_Picture
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.text.paragraph import Paragraph
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def extract_docx_content(docx_path, output_folder):
    document = Document(docx_path)
    os.makedirs(output_folder, exist_ok=True)
    
    markdown_content = []
    image_counter = 0

    def iter_block_items(parent):
        """
        Generate a sequence of block-level items in the document,
        preserving the document structure.
        """
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            raise ValueError("Something's not right")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def process_paragraph(paragraph):
        nonlocal image_counter
        if paragraph.style.name.startswith('Heading'):
            level = int(paragraph.style.name.split()[-1])
            return f"{'#' * level} {paragraph.text}\n\n"
        
        # Check for images in the paragraph
        for run in paragraph.runs:
            for inline in run._element.findall('.//wp:inline', {'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}):
                image_counter += 1
                rId = inline.find('.//a:blip', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}).get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                image_part = document.part.related_parts[rId]
                image_filename = f'image_{image_counter}.png'
                image_path = os.path.join(output_folder, image_filename)
                with open(image_path, 'wb') as f:
                    f.write(image_part.blob)
                return f"![Image {image_counter}]({image_filename})\n\n"
        
        return paragraph.text + "\n\n"

    def process_table(table):
        markdown_table = []
        for i, row in enumerate(table.rows):
            markdown_row = "|" + "|".join(cell.text.strip() for cell in row.cells) + "|"
            markdown_table.append(markdown_row)
            if i == 0:
                markdown_table.append("|" + "|".join("---" for _ in row.cells) + "|")
        return "\n".join(markdown_table) + "\n\n"

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            markdown_content.append(process_paragraph(block))
        elif isinstance(block, Table):
            markdown_content.append(process_table(block))

    markdown_file_path = os.path.join(output_folder, 'output.md')
    with open(markdown_file_path, 'w', encoding='utf-8') as f:
        f.write(''.join(markdown_content))

    return markdown_file_path


def extract_docx_content_in_chunks(docx_path, output_folder):
    document = Document(docx_path)
    os.makedirs(output_folder, exist_ok=True)
    
    markdown_content = []
    image_counter = 0

    def iter_block_items(parent):
        """
        Generate a sequence of block-level items in the document,
        preserving the document structure.
        """
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            raise ValueError("Something's not right")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def process_paragraph(paragraph):
        nonlocal image_counter
        if paragraph.style.name.startswith('Heading'):
            level = int(paragraph.style.name.split()[-1])
            return f"{'#' * level} {paragraph.text}\n\n"
        
        # Check for images in the paragraph
        for run in paragraph.runs:
            for inline in run._element.findall('.//wp:inline', {'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}):
                image_counter += 1
                rId = inline.find('.//a:blip', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}).get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                image_part = document.part.related_parts[rId]
                image_filename = f'image_{image_counter}.png'
                image_path = os.path.join(output_folder, image_filename)
                with open(image_path, 'wb') as f:
                    f.write(image_part.blob)
                return f"![Image {image_counter}]({image_filename})\n\n"
        
        return paragraph.text + "\n\n"

    def process_table(table):
        markdown_table = []
        for i, row in enumerate(table.rows):
            markdown_row = "|" + "|".join(cell.text.strip() for cell in row.cells) + "|"
            markdown_table.append(markdown_row)
            if i == 0:
                markdown_table.append("|" + "|".join("---" for _ in row.cells) + "|")
        return "\n".join(markdown_table) + "\n\n"

    def generate_markdown_chunks(markdown_content, chunk_size=2500, chunk_overlap=250):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        # MD splits
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(''.join(markdown_content))

        # Char-level splits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(md_header_splits)
        for i, doc in enumerate(splits):
            with open(os.path.join(output_folder, f"chunk_{i+1}.md"), 'w', encoding='utf-8') as f:
                f.write(doc.page_content)

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            markdown_content.append(process_paragraph(block))
        elif isinstance(block, Table):
            markdown_content.append(process_table(block))

    generate_markdown_chunks(markdown_content)


if __name__ == "__main__":
    docx_path = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs/BankingRequirements.docx"  # Replace with your input docx file path
    #docx_path = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs/appendix_6m_tool_requirement_specifications_v2_project_portfolio_management_tool.docx"  # Replace with your input docx file path
    output_folder = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs/output"
    
    #markdown_file = extract_docx_content(docx_path, output_folder)
    markdown_file = extract_docx_content_in_chunks(docx_path, output_folder)
    print(f"Markdown file generated: {markdown_file}")