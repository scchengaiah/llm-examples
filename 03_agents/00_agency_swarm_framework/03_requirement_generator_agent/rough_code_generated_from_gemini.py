import docx
import os
import uuid
from PIL import Image

def process_paragraph(paragraph):
  style = paragraph.style.name
  if 'Heading' in style:
    level = int(style.replace('Heading', ''))
    return f"{'#' * level} {paragraph.text}\n"
  else:
    return f"{paragraph.text}\n"

def process_table(table):
  markdown_table = ""
  # Simple table conversion, can be improved for complex tables
  for row in table.rows:
    row_cells = [cell.text for cell in row.cells]
    markdown_table += "| " + " | ".join(row_cells) + " |\n"
    if row == table.rows[0]:
      markdown_table += "| " + "-" * len(row_cells) + " |\n"
  return markdown_table + "\n"

def save_image(image, output_folder):
  image_path = os.path.join(output_folder, f"{uuid.uuid4()}.png")
  with open(image_path, "wb") as f:
    f.write(image.blob)
  return image_path

def parse_docx(input_file, output_folder):
  doc = docx.Document(input_file)
  os.makedirs(output_folder, exist_ok=True)

  markdown_content = ""
  image_counter = 0

  for element in doc.element.body:
    if getattr(element.tag, "name", None) == "p":
      paragraph = element.text
      markdown_content += process_paragraph(paragraph)
    elif getattr(element.tag, "name", None) == "tbl":
      table = element
      markdown_content += process_table(table)
    elif getattr(element.tag, "name", None) == "drawing":
      image = element.graphic.graphicData
      image_path = save_image(image, output_folder)
      markdown_content += f"![Image {image_counter}]({image_path})\n"
      image_counter += 1

  with open(os.path.join(output_folder, "output.md"), "w", encoding='utf-8') as f:
    f.write(markdown_content)

if __name__ == "__main__":
  input_file = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs/appendix_6m_tool_requirement_specifications_v2_project_portfolio_management_tool.docx"  # Replace with your input docx file path
  output_folder = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs"
  parse_docx(input_file, output_folder)
  print("Conversion completed.")
