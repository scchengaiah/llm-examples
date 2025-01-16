# Langgraph Course

This document contains notes captured while going through the Langchain academy's [Langgraph course](https://academy.langchain.com/courses/intro-to-langgraph).

The key aspects involved in the notes may undergo significant upgrades as the framework evolves, hence always refer to the latest documentation for updated information.

We have forked the [repository](https://github.com/langchain-ai/langchain-academy) that contains the sample source code and is segregated into various modules.

All our exploration during the course is performed on `learning` branch.

## My Personal Notes

Langgraph is a low level framework to develop complex agentic applications. It is opensource and provide flexibility to perform extensive customization for complex scenarios.

Langgraph has inbuilt abstractions for state such as MessagesState (`from langgraph.graph import MessagesState`) that can be used by nodes.

Langgraph has inbuilt abstractions to perform tool call and execute the flow conditionally by using `from langgraph.prebuilt import tools_condition` and `from langgraph.prebuilt import ToolNode`.

## State Schema

Schema can be defined as `TypedDict`, `Dataclass` or `Pydantic` where `Pydantic` offers the best approach for runtime validations. Go with `Pydantic` while defining the state.

Refer to [state-schema.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/state-schema.ipynb) file for more information.

## State Reducers

By default, state gets overriden during each execution of node, how can reducers help in managing the state such as appending information using inbuilt operators or custom reducers can be found in the file [state-reducers.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/state-reducers.ipynb).

## Trimming/Filtering of Messages

Langgraph and Langchain together offers several strategies to filter or trim messages to handle the context length of the LLM's. Refer [trim-filter-messages.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/trim-filter-messages.ipynb) for detailed examples.

Refer Langchain documentation on message trimming [here](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens).

## Chatbot Example with summarization and External Memory.

Nice Examples explaining message summarization and persistence can be found in the below links.

[chatbot-summarization.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/chatbot-summarization.ipynb)

[chatbot-external-memory.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-2/chatbot-external-memory.ipynb)

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



### Deployment

Langgraph provides a SDK ([langgraph_sdk](https://pypi.org/project/langgraph-sdk/)) to access the graph deployed on cloud in Langsmith or access via URL exposed by Langgraph studio (can run locally) that is currently supported on MacOS. Refer to [deployment.ipynb](https://github.com/scchengaiah/langchain-academy/blob/main/module-1/deployment.ipynb) file to get a high level idea.

**Opensource possibilities:**

First thought is to use FastAPI application that can expose the graph over an endpoint. Need to check for better alternatives for seamless native support.

For detailed information on this topic, Refer [here](https://langchain-ai.github.io/langgraph/tutorials/deployment/).

For self-hosting with docker, Refer [here](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/).

## References

[Course Source Code](https://github.com/langchain-ai/langchain-academy)

