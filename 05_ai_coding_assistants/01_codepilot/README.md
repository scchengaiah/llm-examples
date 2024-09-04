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
