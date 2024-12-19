# Model Context Protocol (MCP)

The Model Context Protocol is an open standard that enables developers to build secure, two-way connections between their data sources and AI-powered tools. The architecture is straightforward: developers can either expose their data through MCP servers or build AI applications (MCP clients) that connect to these servers.

Currently, the MCP servers are exposed via stdio mode and runs locally that can connect to the clients that also runs locally. As it evolves, we can expect the HTTP mode of communication where servers and clients can be hosted on seperate machines and can be communicated over a network (HTTP).

## Exploration

MCP has community made servers that can be connected by several clients. Some of the well known clients at the moment are `cline(Autonomous Coding Agent)`, `continue(Open source AI coding Assistant)` and `Claude Desktop app`. For more supported clients, refer [here](https://modelcontextprotocol.io/clients).

In this exploration, we shall create our own MCP server and client and establish communication between them. However, we can quickly used community created servers from [here](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#-reference-servers) and can connect to existing clients such as `Claude Desktop app` and utilize the capabilities. For example, Leveraging existing `Brave Search` server and connect to existing local client such as `Claude Desktop app`.

### Setup 1 - Local created MCP server with Local MCP client

We followed the [Quickstart](https://modelcontextprotocol.io/quickstart) setup to perform this exploration.

The locally created MCP server can be found [here](./mcp-server/).

The locally created MCP client can be found [here](./mcp-client/).


## References

[MCP from Anthropic](https://www.anthropic.com/news/model-context-protocol)

[MCP - Docs](https://modelcontextprotocol.io/introduction)