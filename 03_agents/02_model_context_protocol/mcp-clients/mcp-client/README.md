# MCP client

This MCP client connects to the weather server.

## Pre-requisities

To install uv - `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Update path - `SET PATH = %PATH%;C:\Users\20092\.local\bin`

To init new projects and setup dependencies - `uv init <project-name>` and  `cd <project-name>` and `uv venv` and `.venv\Scripts\activate` and `uv add mcp anthropic httpx python-dotenv`

For already available project, create virtual environment - `uv venv` and `.venv\Scripts\activate`

Sync dependencies for existing projects - `uv sync`

To launch the client with the already created weather server, launch the project directory and execute `uv run client.py ../../mcp-servers/weather/src/weather/server.py`.

**Important Note:** It should be noted that the python installation must contain all dependencies required in order for the server and client to work together. In our case, we have used the python command from the virtual env that contains all required dependencies for both the server and the client.