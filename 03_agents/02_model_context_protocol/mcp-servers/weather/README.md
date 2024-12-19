# Weather Server

## Pre-requisities

To install uv - `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Update path - `SET PATH = %PATH%;C:\Users\20092\.local\bin`

To init new projects and setup dependencies - `uv init <project-name>` and  `cd <project-name>` and `uv venv` and `.venv\Scripts\activate` and `uv add mcp httpx`

For already available project, create virtual environment - `uv venv` and `.venv\Scripts\activate`

Sync dependencies for existing projects - `uv sync`

To run the server, launch the project directory and execute `uv run src/weather/server.py`