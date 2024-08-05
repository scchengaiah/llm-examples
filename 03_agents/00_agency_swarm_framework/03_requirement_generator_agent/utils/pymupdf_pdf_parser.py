import pymupdf4llm
import pathlib
import os

FILE_PATH = "D:/gitlab/learnings/artificial-intelligence/llm-examples/03_agents/00_agency_swarm_framework/03_requirement_generator_agent/resources/Scale_NCA_RFP_PLM.pdf"

FILE_PATH = "D:/tmp/appendix_6m_tool_requirement_specifications_v2_project_portfolio_management_tool.pdf"

FILE_PATH = "C:/Users/20092/Documents/Temp/pdftoocr/pngtopdf_2.pdf"

MARKDOWN_PATH = "D:/tmp/pymupdf4llm"

os.makedirs(MARKDOWN_PATH, exist_ok=True)

md_text = pymupdf4llm.to_markdown(FILE_PATH, write_images=True, page_chunks=False, image_path=MARKDOWN_PATH)

pathlib.Path(f"{MARKDOWN_PATH}/output.md").write_bytes(md_text.encode())