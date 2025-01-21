# Langgraph Course

This document contains notes captured while going through the Langchain academy's [Langgraph course](https://academy.langchain.com/courses/intro-to-langgraph).

The key aspects involved in the notes may undergo significant upgrades as the framework evolves, hence always refer to the latest documentation for updated information.

We have forked the [repository](https://github.com/langchain-ai/langchain-academy) that contains the sample source code and is segregated into various modules.

All our exploration during the course is performed on `learning` branch.

## My Personal Notes

Langgraph is a low level framework to develop complex agentic applications. It is opensource and provide flexibility to perform extensive customization for complex scenarios.

Langgraph has inbuilt abstractions for state such as MessagesState (`from langgraph.graph import MessagesState`) that can be used by nodes.

Langgraph has inbuilt abstractions to perform tool call and execute the flow conditionally by using `from langgraph.prebuilt import tools_condition` and `from langgraph.prebuilt import ToolNode`.

Langgraph Studio offers an excellent UI way for testing graphs, however, it is supported locally only for macos users. For other platforms, we have a workaround by running the local langgraph server and launch it via langsmith. This [documentation](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#development-server-with-web-ui) provides clear instructions to perform the same.

### State Schema

Schema can be defined as `TypedDict`, `Dataclass` or `Pydantic` where `Pydantic` offers the best approach for runtime validations. Go with `Pydantic` while defining the state.

Refer to [state-schema.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/state-schema.ipynb) file for more information.

### State Reducers

By default, state gets overriden during each execution of node, how can reducers help in managing the state such as appending information using inbuilt operators or custom reducers can be found in the file [state-reducers.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/state-reducers.ipynb).

### Trimming/Filtering of Messages

Langgraph and Langchain together offers several strategies to filter or trim messages to handle the context length of the LLM's. Refer [trim-filter-messages.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/trim-filter-messages.ipynb) for detailed examples.

Refer Langchain documentation on message trimming [here](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens).

### Persistence/Memory

Langgraph has a concept of checkpoints that are collectively represented as threads that provide memory to the agent. Refer to the example [agent-memory.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-1/agent-memory.ipynb).

The in-memory solution can be found here `from langgraph.checkpoint.memory import MemorySaver`. An example implementation is shown below. 

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

To scale in production scenarios, Langchain provides open-sourced `langgraph_checkpoint_sqlite` and `langgraph_checkpoint_postgres` implementations. We can use `langgraph_checkpoint_postgres` for production based workloads. More information can be found [here](https://github.com/langchain-ai/langgraph/releases/tag/0.2.0). 

Implementation example with `postgres` can be found [here](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)

#### Chatbot Example with summarization and External Memory.

Nice Examples explaining message summarization and persistence can be found in the below links.

[chatbot-summarization.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/chatbot-summarization.ipynb)

[chatbot-external-memory.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/chatbot-external-memory.ipynb)

### Streaming

Langgraph offers some advanced streaming patterns that helps to stream the graph state, messages, tool calls.

Refer to the file [streaming-interruption.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-3/streaming-interruption.ipynb) for more information.

> It is important to note that the approach of streaming when the langgraph agents are exposed via API is different. This is covered in the above example using langgraph studio. However, when we deploy our langgraph application as backend API for the usage of langgraph API using self-hosted deployment, we can leverage similar API calls to perform streaming.

### Interrupts - Update State, Human in the Feedback loop and Time Travel concepts

More information can be found [here](https://github.com/scchengaiah/langchain-academy/tree/main/module-3).

### Deployment

Langgraph provides a SDK ([langgraph_sdk](https://pypi.org/project/langgraph-sdk/)) to access the graph deployed on cloud in Langsmith or access via URL exposed by Langgraph studio (can run locally) that is currently supported on MacOS. Refer to [deployment.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-1/deployment.ipynb) file to get a high level idea.

**Opensource possibilities:**

First thought is to use FastAPI application that can expose the graph over an endpoint. Need to check for better alternatives for seamless native support.

For detailed information on this topic, Refer [here](https://langchain-ai.github.io/langgraph/tutorials/deployment/).

For self-hosting with docker, Refer [here](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/).

### Research Assistant

For a fully functional example of research assistant, Refer [research-assistant.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-4/research-assistant.ipynb).

To dive deeper and explore the Multi Agent architecture, Refer [here](https://langchain-ai.github.io/langgraph/tutorials/#agent-architectures).

## References

[Course Source Code](https://github.com/langchain-ai/langchain-academy)

[LangGraph Core Concepts](https://dev.to/jamesli/introduction-to-langgraph-core-concepts-and-basic-components-5bak)

