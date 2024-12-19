# Model Context Protocol (MCP)

The Model Context Protocol is an open standard that enables developers to build secure, two-way connections between their data sources and AI-powered tools. The architecture is straightforward: developers can either expose their data through MCP servers or build AI applications (MCP clients) that connect to these servers.

Currently, the MCP servers are exposed via stdio mode and runs locally that can connect to the clients that also runs locally. As it evolves, we can expect the HTTP mode of communication where servers and clients can be hosted on seperate machines and can be communicated over a network (HTTP).

## Exploration

MCP has community made servers that can be connected by several clients. Some of the well known clients at the moment are `cline(Autonomous Coding Agent)`, `continue(Open source AI coding Assistant)` and `Claude Desktop app`. For more supported clients, refer [here](https://modelcontextprotocol.io/clients).

In this exploration, we shall create our own MCP server and client and establish communication between them. However, we can quickly used community created servers from [here](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#-reference-servers) and can connect to existing clients such as `Claude Desktop app` and utilize the capabilities. For example, Leveraging existing `Brave Search` server and connect to existing local client such as `Claude Desktop app`.

### Setup 1 - Local created MCP server with Local MCP client

We followed the [Quickstart](https://modelcontextprotocol.io/quickstart) setup to perform this exploration.

The locally created MCP server can be found [here](./mcp-servers/). Refer to README.md in respective server folders.

The locally created MCP client can be found [here](./mcp-clients/). Refer to README.md in respective client folders.

To connect via MCP inspector client. For this follow the below steps:

Related documentation: https://modelcontextprotocol.io/docs/tools/inspector

Launch MCP inspector: `npx @modelcontextprotocol/inspector uv --directory D:/gitlab/learnings/artificial-intelligence/llm-examples/03_agents/02_model_context_protocol/mcp-servers/weather/src/weather run weather`

Above command launches a web interface `http://localhost:5173` and clicking on `Connect` establishes connection to the server and we can list the tools.

### Setup 2 - Using Claude Desktop

For this setup, we install `Claude Desktop` app and going to leverage existing community server [Postgres](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) that shall be used to fetch the latest information and augment the response via language model.

Create configuration file `claude_desktop_config.json` in the directory `%APPDATA%/Claude`.

Add the following content in the `claude_desktop_config.json`.

```json
{
    "mcpServers": {
        "postgres": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-postgres",
            "postgresql://user:password@localhost:5432/mydatabase"
        ]
        }
    }
}
```

We can start using the attached Postgres server.

Note that this is currently working for `Claude Desktop app` that contains professional plan.

### Setup 3 - Using Cline

Cline (Autonomous coding agent) tool has MCP servers concept integrated into it. We have included the existing postgres server for testing purposes.

Below is the json used, Had to perform certain workaround to overcome limitations in windows when using `npx`, hence we installed the modules in advance and used it.

This solution was proposed in this [link](https://github.com/modelcontextprotocol/servers/issues/75). Refer to this more details.

```json
{
    "mcpServers": {
        "postgres": {
        "command": "C:/Program Files/nodejs/node.exe",
        "args": [
            "C:/Users/20092/AppData/Roaming/npm/node_modules/@modelcontextprotocol/server-postgres/dist/index.js",
            "postgresql://username:password@localhost:5432/database"
        ]
        }
    }
}
```

## References

[MCP from Anthropic](https://www.anthropic.com/news/model-context-protocol)

[MCP - Docs](https://modelcontextprotocol.io/introduction)