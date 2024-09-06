import ast
import json
import os
import argparse
from typing import Dict, List, Any, Optional
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

class CodeVisitor(ast.NodeVisitor):
    """
    A custom AST visitor that traverses the abstract syntax tree of Python code
    and collects metadata about classes and functions.
    """

    def __init__(self, filename: str, source_lines: List[str], project_root: str):
        """
        Initialize the CodeVisitor.

        Args:
            filename (str): The name of the file being processed.
            source_lines (List[str]): The source code lines of the file.
            project_root (str): The root directory of the project.
        """
        self.filename = filename
        self.source_lines = source_lines
        self.project_root = project_root
        self.current_class = None
        self.current_function = None
        self.items = []

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit a class definition in the AST.

        Args:
            node (ast.ClassDef): The class definition node.
        """
        self.current_class = node.name
        class_item = self.create_item(node, "Class")
        self.items.append(class_item)
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit a function definition in the AST.

        Args:
            node (ast.FunctionDef): The function definition node.
        """
        self.current_function = node.name
        func_item = self.create_item(node, "Function")
        self.items.append(func_item)
        self.generic_visit(node)
        self.current_function = None

    def create_item(self, node: ast.AST, code_type: str) -> Dict[str, Any]:
        """
        Create a metadata item for a class or function.

        Args:
            node (ast.AST): The AST node representing the class or function.
            code_type (str): The type of code item ("Class" or "Function").

        Returns:
            Dict[str, Any]: A dictionary containing metadata about the code item.
        """
        docstring = ast.get_docstring(node)
        signature = self.get_signature(node)
        snippet = self.get_source_from_node(node)

        # Calculate relative path
        relative_path = os.path.relpath(self.filename, self.project_root)
        # Ensure forward slashes
        relative_path = relative_path.replace(os.sep, '/')

        return {
            "name": node.name,
            "signature": signature,
            "code_type": code_type,
            "docstring": docstring,
            "line": node.lineno,
            "line_from": node.lineno,
            "line_to": node.end_lineno,
            "context": {
                "module": self.get_module_name(),
                "file_path": relative_path,
                "file_name": os.path.basename(self.filename),
                "class_name": self.current_class,
                "function_name": self.current_function if code_type != "Function" else None,
                "snippet": snippet
            }
        }

    def get_signature(self, node: ast.AST) -> str:
        """
        Extract the signature (declaration) of a class or function.

        Args:
            node (ast.AST): The AST node representing the class or function.

        Returns:
            str: The signature of the class or function.
        """
        if isinstance(node, ast.FunctionDef):
            args = []
            defaults = list(map(ast.unparse, node.args.defaults))
            for i, arg in enumerate(node.args.args):
                if i >= len(node.args.args) - len(defaults):
                    default = defaults[i - (len(node.args.args) - len(defaults))]
                    args.append(f"{arg.arg}={default}")
                else:
                    args.append(arg.arg)
            
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")
            
            return f"def {node.name}({', '.join(args)}):"
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            return f"class {node.name}({bases}):"
        else:
            return ""

    def get_source_from_node(self, node: ast.AST) -> str:
        """
        Extract the complete source code for a given AST node.

        Args:
            node (ast.AST): The AST node to extract source code from.

        Returns:
            str: The source code of the node.
        """
        return "".join(self.source_lines[node.lineno - 1:node.end_lineno]).strip()

    def get_module_name(self) -> str:
        """
        Get the module name from the filename.

        Returns:
            str: The module name.
        """
        return os.path.splitext(os.path.basename(self.filename))[0]

def parse_python_file(file_path: str, project_root: str) -> List[Dict[str, Any]]:
    """
    Parse a Python file and extract metadata about classes and functions.

    Args:
        file_path (str): The path to the Python file.
        project_root (str): The root directory of the project.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing metadata about classes and functions.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        source_lines = content.splitlines()
    
    tree = ast.parse(content)
    visitor = CodeVisitor(file_path, source_lines, project_root)
    visitor.visit(tree)
    return visitor.items

def get_ignore_spec(project_root: str, custom_ignore_file: Optional[str] = None) -> PathSpec:
    """
    Read the .gitignore file and optional custom ignore file to create a PathSpec object.

    Args:
        project_root (str): The root directory of the project.
        custom_ignore_file (Optional[str]): Path to a custom ignore file.

    Returns:
        PathSpec: A PathSpec object representing the combined ignore patterns.
    """
    ignore_patterns = []

    # Read .gitignore file
    gitignore_path = os.path.join(project_root, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding="utf-8") as gitignore_file:
            ignore_patterns.extend(gitignore_file.read().splitlines())

    # Read custom ignore file if provided
    if custom_ignore_file and os.path.exists(custom_ignore_file):
        with open(custom_ignore_file, 'r', encoding="utf-8") as custom_file:
            ignore_patterns.extend(custom_file.read().splitlines())

    return PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)

def process_project(project_root: str, output_file: str, custom_ignore_file: Optional[str] = None):
    """
    Process all Python files in a project, extract metadata, and write it to a JSONL file.

    Args:
        project_root (str): The root directory of the project.
        output_file (str): The path to the output JSONL file.
        custom_ignore_file (Optional[str]): Path to a custom ignore file.
    """
    ignore_spec = get_ignore_spec(project_root, custom_ignore_file)
    
    with open(output_file, 'w', encoding="utf-8") as outfile:
        for root, _, files in os.walk(project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, project_root)
                    
                    if ignore_spec.match_file(relative_path):
                        print(f"Skipping ignored file: {relative_path}")
                        continue
                    
                    print(f"Processing file: {relative_path}")
                    try:
                        items = parse_python_file(file_path, project_root)
                        for item in items:
                            json.dump(item, outfile)
                            outfile.write('\n')
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Python files in a project and generate metadata.")
    parser.add_argument("--project-root", help="Root directory of the project to parse")
    parser.add_argument("--output-file", help="Path to the output JSONL file")
    parser.add_argument("--ignore-file", help="Path to a custom ignore file (in addition to .gitignore)")
    
    args = parser.parse_args()
    
    process_project(args.project_root, args.output_file, args.ignore_file)
    print(f"Parsing completed. Output written to {args.output_file}")