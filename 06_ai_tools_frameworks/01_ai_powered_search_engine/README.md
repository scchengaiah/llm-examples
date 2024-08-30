# AI Powered Search Engines

Here, we will explore and document the findings with respect to AI powered search engines (Open source + Commercial) and setup for further exploration.

## Search Engines

AI powered search engines that can navigate through the internet to fetch relevant results.

### Perplexica

[Perplexica](https://github.com/ItzCrazyKns/Perplexica) is an open-source AI-powered searching tool or an AI-powered search engine that goes deep into the internet to find answers. Inspired by Perplexity AI, it's an open-source option that not just searches the web but understands your questions. It uses advanced machine learning algorithms like similarity searching and embeddings to refine results and provides clear answers with sources cited.

Using [SearxNG](https://github.com/searxng/searxng) to stay current and fully open source, Perplexica ensures you always get the most up-to-date information without compromising your privacy.

`SearXNG` is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled.

[searx.space](https://searx.space/) contains list of hosted sites that can offer search services via hosted `SearXNG` .

For more privacy, users can run their own `SearxNG` instance on their laptop with more controlled settings. Refer to the [docs](https://docs.searxng.org/) for more details.

#### Setup

We followed docker based setup as recommended [here](https://github.com/ItzCrazyKns/Perplexica?tab=readme-ov-file#getting-started-with-docker-recommended).

In order to make this work, we had to perform certain changes from the networking perspective. Before applying these changes, the front-end interface was spinning without any functionality.

We followed this [recommendation](https://github.com/ItzCrazyKns/Perplexica/issues/180#issuecomment-2158487861) along with [NETWORKING.md](https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/installation/NETWORKING.md) document from the repository.

### Mindsearch

[MindSearch](https://github.com/InternLM/MindSearch) is an open-source AI Search Engine Framework with Perplexity.ai Pro performance. You can simply deploy it with your own perplexity.ai style search engine with either close-source LLMs (GPT, Claude) or open-source LLMs (InternLM2.5 series are specifically optimized to provide superior performance within the MindSearch framework; other open-source models have not been specifically tested).

This claims that the results retrieved are efficient compared to other platforms available in the market such as Perplexity.

Let us explore by setting this up and evaluate its capabilities.

### Setup

> WORK IN PROGRESS.

## Document based Search Engines

AI powered document based search engines that can respond to user queries in natural language over documentation.

### Enhance Docs

[Enhance Docs](https://docs.enhancedocs.com/) is an open-source AI-powered search engine specifically designed for documentation. It leverages AI algorithms to improve the search experience by understanding the context and intent behind queries, making it suitable for technical documentation and knowledge bases.

We can ingest our documents(Github, Existing documentation sites based on Docusaurus, etc) and ask questions or chat using enhance docs powered interface.

## References

[Perplexity AI - Commercial](https://www.perplexity.ai/)
