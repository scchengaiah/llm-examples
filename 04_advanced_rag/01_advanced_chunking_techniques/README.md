## References

[Mastering RAG - Advanced Chunking Techniques](https://www.rungalileo.io/blog/mastering-rag-advanced-chunking-techniques-for-llm-applications#recursive-character-splitter)

[Langchain - How to load PDF ?](https://python.langchain.com/docs/how_to/document_loader_pdf/)

[Langchain - Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)

[Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/tree/main/tutorials/LevelsOfTextSplitting) - [Local copy of notebook](./01_5_Levels_Of_Text_Splitting.ipynb)
### Semantic Splitting

[Example Source](./02_semantic_splitting/)

[A guide to understand Semantic Splitting for document chunking in LLM applications](https://www.reddit.com/r/LangChain/comments/1erxo60/a_guide_to_understand_semantic_splitting_for/)

[Semantic Splitting From Scratch](https://www.youtube.com/watch?v=qvDbOYz6U24) - [Related Github Repo](https://github.com/bitswired/semantic-splitting-tutorial)

### Contextual Embeddings

[New technique makes RAG systems much better at retrieving the right documents](https://venturebeat.com/ai/new-technique-makes-rag-systems-much-better-at-retrieving-the-right-documents/) - [Hugging Face Link for the Contextual embedding model](https://huggingface.co/jxm/cde-small-v1)

### Proposition-Based Retrieval

[Example Source](./03_llm_based_chunking/)

Complete Langchain template implementation can be found here. This implementation leverages the concept of creating `Propositions` (Standalone representations of a given text) using LLM, followed by leveraging [MultiVectorRetriever](https://python.langchain.com/docs/how_to/multi_vector/) approach where in the `Propositions` are stored in a vector store for semantic retrieval and larger portion of the document involved in creating these `Propositions` are stored within a doc store.

How does this work ?

When a user asks a query, the `MultiVectorRetriever` hits the vectorstore to fetch relevant documents and for each of these documents, the parent document (Larger context) is fetched and sent to the LLM as context along with the user query.

[Proposition based retriever Langchain Template](https://github.com/langchain-ai/langchain/tree/master/templates/propositional-retrieval)