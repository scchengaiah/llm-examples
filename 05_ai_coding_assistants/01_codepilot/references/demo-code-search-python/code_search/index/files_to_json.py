import os.path
import json
from pathlib import Path
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

from code_search.config import DATA_DIR

def process_file(root_dir, file_path):
    """
    Processes a single file to extract its relative path and code lines.

    Args:
        root_dir (str): The root directory from which relative paths are calculated.
        file_path (str): The path of the file to process.

    Returns:
        dict: A dictionary containing the relative path, code lines, start line, and end line.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        code_lines = file.readlines()
        relative_path = os.path.relpath(file_path, root_dir)
        # Ensure forward slashes
        relative_path = relative_path.replace(os.sep, '/')
        return {
            "path": relative_path,
            "code": code_lines,
            "startline": 1,
            "endline": len(code_lines)
        }

def get_ignore_spec(project_root: str) -> PathSpec:
    """
    Read the .gitignore file and optional custom ignore file to create a PathSpec object.

    Args:
        project_root (str): The root directory of the project.
    Returns:
        PathSpec: A PathSpec object representing the combined ignore patterns.
    """
    ignore_patterns = []

    # Read .gitignore file
    gitignore_path = os.path.join(project_root, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding="utf-8") as gitignore_file:
            ignore_patterns.extend(gitignore_file.read().splitlines())

    return PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)


def explore_directory(root_dir):
    """
    Explores the directory tree starting from the root directory and processes all .rs files.

    Args:
        root_dir (str): The root directory to start exploring.

    Returns:
        list: A list of dictionaries, each containing data of a processed .rs file.
    """
    ignore_spec = get_ignore_spec(root_dir)
    result = []
    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            relative_path = os.path.relpath(file_path, root_dir)
            if file_path.endswith('.py'):
                if ignore_spec.match_file(relative_path):
                    print(f"Skipping ignored file: {relative_path}")
                    continue
                print(f"Processing file: {relative_path}")
                result.append(process_file(root_dir, file_path))
    return result

def main():
    """
    Main function that sets the folder path, explores the directory, and writes the processed data to a JSON file.

    The folder path is retrieved from the REPO_PATH environment variable, and the output file is saved
    to the DATA_DIR directory with the name 'rs_files.json'.
    """
    folder_path = os.getenv('REPO_PATH')
    output_file = Path(DATA_DIR) / "py_files.json"

    files_data = explore_directory(folder_path)

    if not os.path.exists(Path(DATA_DIR)):
        os.makedirs(Path(DATA_DIR))

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(files_data, json_file, indent=2)

if __name__ == "__main__":
    main()