import docx

def extract_tables_from_docx(docx_path):
    doc = docx.Document(docx_path)
    tables = []
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        tables.append(table_data)
    return tables

def table_to_markdown(table):
    if not table:
        return ""

    # Find the maximum number of columns
    max_cols = max(len(row) for row in table)

    # Pad rows with empty strings if they have fewer cells
    padded_table = [row + [''] * (max_cols - len(row)) for row in table]

    # Calculate the maximum width of each column
    col_widths = [max(len(str(row[i])) for row in padded_table) for i in range(max_cols)]

    # Create the header row
    header = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(padded_table[0])) + " |"
    separator = "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"

    # Create the data rows
    rows = []
    for row in padded_table[1:]:
        rows.append("| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")

    # Combine all parts
    return "\n".join([header, separator] + rows)

def main(docx_path, output_path):
    tables = extract_tables_from_docx(docx_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, table in enumerate(tables):
            f.write(f"Table {i+1}:\n\n")
            f.write(table_to_markdown(table))
            f.write("\n\n")

    print(f"Markdown tables have been written to {output_path}")

if __name__ == "__main__":
    docx_path = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs/appendix_6m_tool_requirement_specifications_v2_project_portfolio_management_tool.docx"  # Replace with your input docx file path
    output_folder = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs/output"
    output_path = "D:/OneDrive - IZ/MxIoT-Documents/Projects/GenAI-Intelizign/RequirementGenerator/ExampleDocs/output/markdown_tables.md"
    main(docx_path, output_path)