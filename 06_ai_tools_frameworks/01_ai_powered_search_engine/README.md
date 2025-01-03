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

#### Docker based

[Reference](https://github.com/InternLM/MindSearch/tree/main/docker#mindsearch-docker-compose-user-guide)

#### Manual based

We try manual based setup in order to make some changes to the source code in case of any unique requirments.

We follow this [documentation](https://github.com/InternLM/MindSearch/tree/main?tab=readme-ov-file#%EF%B8%8F-build-your-own-mindsearch) to perform the setup.

We have cloned the repo to our local machine and applied few changes. Mindsearch heavily relies on [LMDeploy - toolkit for compressing, deploying, and serving LLMs.](https://github.com/InternLM/lmdeploy) that can serve LLMs and also supports OpenAI compatible interface related endpoints via [LAgent - A lightweight framework for building LLM-based agents](https://github.com/InternLM/lagent) framework.

We have copied the [GPTAPI](https://github.com/InternLM/lagent/blob/main/lagent/llms/openai.py) implementation from `LAgent` and modified to support `Azure OpenAI` endpoint. This implementation can be found [here](./MindSearch/mindsearch/agent/azure_openai.py).

Similarly, we can setup custom implementation for other providers such as `Amazon Bedrock` to leverage Anthropic `Claude` models.

[azure_openai.py](./MindSearch/mindsearch/agent/azure_openai.py) is a custom version created extending `BaseAPIModel` from `lagent` library.

After installing the requirements via `pip install -r requirements.txt` Check [run.sh](./MindSearch/run.sh) that contains command to start Mindsearch.

For frontend, we have `streamlit`, `gradio`, and `react` as options. To run with streamlit launch another command prompt and execute `streamlit run frontend/mindsearch_streamlit.py`. For other frontend interfaces, refer to the documentation.

## Document based Search Engines

AI powered document based search engines that can respond to user queries in natural language over documentation.

### Enhance Docs

[Enhance Docs](https://docs.enhancedocs.com/) is an open-source AI-powered search engine specifically designed for documentation. It leverages AI algorithms to improve the search experience by understanding the context and intent behind queries, making it suitable for technical documentation and knowledge bases.

We can ingest our documents(Github, Existing documentation sites based on Docusaurus, etc) and ask questions or chat using enhance docs powered interface.

## References

[Perplexity AI - Commercial](https://www.perplexity.ai/)
