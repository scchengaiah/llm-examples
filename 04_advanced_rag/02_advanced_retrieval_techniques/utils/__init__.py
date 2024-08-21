import re
import json

def remove_trailing_forward_slash(url):
    if url.endswith("/"):
        return url[:-1]
    return url

def add_trailing_forward_slash(url):
    if not url.endswith("/"):
        return url + "/"
    return url

def extract_json_from_markdown(markdown_text):
    """Extract json content from markdown representation

    Example markdown text:
        ```json
        {
            "name": "John",
            "age": 30
        }
        ```

    Args:
        markdown_text (_type_): Markdown json content

    Raises:
        ValueError: Invalid JSON content

    Returns:
        json: Parsed JSON content
    """
    # Regular expression to match JSON content within ```json blocks
    json_pattern = r'```json\s*([\s\S]*?)\s*```'

    # Find all matches
    matches = re.findall(json_pattern, markdown_text)

    if not matches:
        raise ValueError("No JSON content found in the markdown text")

    # Take the first match (assuming there's only one JSON block)
    json_string = matches[0]

    # Parse the JSON string
    try:
        json_data = json.loads(json_string)
        return json_data
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {str(e)}"}

def flatten(nested_list: list) -> list:
    """Flatten a list of lists into a single list."""

    return [item for sublist in nested_list for item in sublist]
