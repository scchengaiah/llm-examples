#!/usr/bin/env bash

set -e

REPO_PATH=$1

REPO_PATH=$(realpath $REPO_PATH)

# Get path to this script
SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ROOT_PATH=$SCRIPT_PATH/..

# Extract all files from REPO_PATH with .py extension and write to json file (py_files.json).
python -m code_search.index.files_to_json

# Read the json file (py_files.json) and upload it to the Qdrant collection named "python-code-files".
python -m code_search.index.file_uploader

# Generate structured representation of the repository and write to jsonl file (py_files_structured.jsonl).
python -m code_search.index.files_to_structured_jsonl

# Read the jsonl file (py_files_structured.jsonl) and upload it to the Qdrant collection named "python-code-snippets-unixcoder".
python -m code_search.index.upload_code

# Read the jsonl file (py_files_structured.jsonl) and upload it to the Qdrant collection named "python-code-signatures".
python -m code_search.index.upload_signatures
