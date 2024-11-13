import os
# Sample data based on your given output
splits = [
    {'metadata': {'header_1': 'Jio Financial Services Limited Overview and Principles'},
     'page_content': '# Jio Financial Services Limited Overview and Principles\nLets say we have some data here.'},
    {'metadata': {'header_1': 'Jio Financial Services Limited Overview and Principles', 'header_2': 'About Jio Financial Services Limited (JFSL)'},
     'page_content': '## About Jio Financial Services Limited (JFSL)'},
    {'metadata': {'header_1': 'Jio Financial Services Limited Overview and Principles', 'header_2': 'About Jio Financial Services Limited (JFSL)'},
     'page_content': 'JFSL was a systemically important non-deposit taking non-banking financial company ("NBFC"), registered with the Reserve Bank of India (RBI)...'},
    {'metadata': {'header_1': 'Jio Financial Services Limited Overview and Principles', 'header_2': 'The 4Rs: Guiding Principles of Our Operations', 'header_3': 'Regulatory Adherence'},
     'page_content': '### Regulatory Adherence\nWe are multi-regulated across our businesses...'}
]

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

# Convert the dictionary-like input to objects with attributes
class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

def concatenate_markdown_files(folder_path, output_file):
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Loop through all files in the folder
        for i, _ in enumerate(folder_path):
            # Increment by one to start from 1
            idx = i+1
            filename = f"page-chunk-output-{idx}.md"  
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):  
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # Write each file's content to the output file
                    outfile.write(infile.read())
                    outfile.write('\n\n')  # Newline between files (optional)

    print(f'All markdown files have been concatenated into {output_file}.')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
temp_dir = os.path.join(project_root, "temp")
chunk_dir= os.path.join(temp_dir, "chunks")
output_file = os.path.join(temp_dir, "consolidated_markdown.md")

# concatenate_markdown_files(chunk_dir, output_file)
# Convert your input format to Document objects
#splits = [Document(split['metadata'], split['page_content']) for split in splits]
#
## Use the function
#final_markdown = reconstruct_markdown(splits)
#print(final_markdown)