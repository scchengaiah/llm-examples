## Docling parses documents and exports them to the desired format with ease and speed.
## https://github.com/DS4SD/docling
## https://ds4sd.github.io/docling/

from docling.document_converter import DocumentConverter
import os
import time


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
image_dir = os.path.join(data_dir, "images")
temp_dir = os.path.join(project_root, "temp")

pdf_file_path = os.path.join(data_dir, "jio-financial-services-annual-report-2023-2024-small.pdf")

print("Instantiating DocumentConverter...")
converter = DocumentConverter()

start_time = time.time()

print("Conversion Started...")
result = converter.convert(pdf_file_path)
print("Exporting to Markdown...")
print(result.document.export_to_markdown())

print(f"Time taken for conversion: {time.time() - start_time}")