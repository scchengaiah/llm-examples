# CodePilot

An AI assisted pair programmer built on top of existing frameworks with Agentic workflows for accuracy.

## Roadmap

- Step 1: To identify reliable solution to query code base semantically. Evaluate the retrieval accuracy and improve further.
  - Agentic workflow to validate the retrieved context and re-iterate with different queryset if required.
- Step 2: To use the retrieved context as reference and apply changes to the code.
  - Agentic workflow to determine changes involved and evaluate frameworks such as `aider` can be utilized to perform the changes.
- Step 3: Evaluate and improve the implementation for optimal accuracy.

## Development Progress

We are currently in research phase trying to understand the nuances of searching over a code base to identify and extract relevant code snippets, files based on user query.

**Research activity draft:**

- Semantic Search over a Codebase

  - Setup a working example with decent accuracy to identify the required information from the existing code base.
  - **References:**

    **Blogs:**

    https://www.greptile.com/blog/semantic - Excellent blog explaining how semantic search works over a codebase.

    https://qdrant.tech/documentation/tutorials/code-search/ - To generate Code base indexes for embeddings:

    - https://github.com/sourcegraph/scip - Generate LSIF Index for various languages
    - https://github.com/sourcegraph/scip-python?tab=readme-ov-file - Generate LSIF Index for python project to proceed with embeddings (text+code) as recommended in the article.
    - https://github.com/qdrant/demo-code-search

    **Repos for Semantic Code Search:**

    https://github.com/sturdy-dev/semantic-code-search - Implemented long back using Transformer based models. Advantage is that it can point to any existing repo with ease and can start answering questions. It uses tree sitter parsing Generators and particularly the one created with python bindings to navigate the source code and generate embeddings.
    [Tree Sitter](https://tree-sitter.github.io/tree-sitter/), [Tree Sitter with Python bindings](https://github.com/grantjenks/py-tree-sitter-languages)

    https://github.com/fynnfluegge/codeqai - Leverages Tree Sitter for parsing, Langchain and AI powered code Search.

    https://github.com/kantord/SeaGOAT - Has Server Client architecture, Uses ChromaDB under the hood for embeddings generation and querying purpose. Needs more time to setup installation. It does iterate through files and chunks based on lines from initial impressions.

    **Others:**

    https://github.com/getzep/zep - Zep: Long-Term Memory for ‍AI Assistants.

    https://github.com/fynnfluegge/doc-comments-ai - Generate documentation for your code using AI. A well documented code can be beneficial to improve the quality of the semantic retrieval.

## Current activities

Based on the above research on [Semantic search over a Codebase](#development-progress), we need to perform the following:

- Explore codeqai and setup locally with chat bot integration.
  - Local setup complete with debugging. In windows check Appdata (Roaming and Local folders) where they are maintaining config and embedding cache.
  - Treesitter was used to generate embeddings into FAISS Database.
  - Works fine for simple usecases, felt the way in which the code was parsed and embedded is not to the level of qdrant example.
  - As part of next step, we can try to generate similar jsonl format as qdrant did for better embeddings of source code.
- Setup qdrant based vector store for semantic code search. - Require parsing library to parse python files and convert into a storable representation before implementation.
  - Able to create an example to parse python source code equivalent to that of rust impln mentioned in the notebook.
  - The impln can be found [here](./codepilot/python_parser.py). This can accomodate the example mentioned in the [notebook](https://colab.research.google.com/github/qdrant/examples/blob/master/code-search/code-search.ipynb), [Local Impln](./codepilot/code_search.ipynb)
  - As part of next step, We have analyzed [demo-code-search](https://github.com/qdrant/demo-code-search) repo that contains end to end advanced impln for the code search for the Rust repository, we have decomposed the end to end activities performed to setup a similar example for python based repositories. The analyzed impln can be found [here](./references/demo-code-search-rust/).
  - Our plan is to create a similar implementation for python based repository and setup a searchable user interface with api backend to proceed with agentic integration.
