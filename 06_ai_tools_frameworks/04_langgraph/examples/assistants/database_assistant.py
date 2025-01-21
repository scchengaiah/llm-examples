# %%
from dotenv import load_dotenv

env_loaded = load_dotenv()
env_loaded

import os

# %%
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_postgres import PGVector

azure_deployment = "gpt-4o-mini"
# azure_deployment = "gpt-4o"

postgres_host = os.getenv("POSTGRES_HOST")
postgres_port = os.getenv("POSTGRES_PORT")
postgres_user = os.getenv("POSTGRES_USER")
postgres_db = os.getenv("POSTGRES_DB")
postgres_password = os.getenv("POSTGRES_PASSWORD")

connection = f"postgresql+psycopg://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=azure_deployment,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
    max_tokens=8192,
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small", api_version="2024-02-01"
)

# %%
## Tools Definition

from typing import Dict, List

import sqlalchemy as sa
from sqlalchemy import MetaData, inspect

schema_info_str = None


def get_database_schema() -> str:
    """
    Query the PostgreSQL database and return a formatted string containing all tables
    and their columns with data types.
    """
    global schema_info_str  # Declare as global to modify it

    if schema_info_str:
        return schema_info_str
    else:
        try:
            # Create engine
            engine = sa.create_engine(connection)
            inspector = inspect(engine)

            # Get all table names
            tables = inspector.get_table_names()

            if not tables:
                return "No tables found in the database."

            # Build formatted output
            schema_info = ["Database Schema:"]
            schema_info.append("=" * 50)

            for table in tables:
                schema_info.append(f"\nTable: {table}")
                schema_info.append("-" * 30)

                # Get columns for each table
                columns = inspector.get_columns(table)
                for column in columns:
                    col_name = column["name"]
                    col_type = str(column["type"])
                    nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
                    primary_key = inspector.get_pk_constraint(table)
                    is_pk = (
                        "PRIMARY KEY"
                        if col_name in primary_key["constrained_columns"]
                        else ""
                    )

                    schema_info.append(
                        f"  - {col_name}: {col_type} {nullable} {is_pk}".rstrip()
                    )

            schema_info_str = "\n".join(schema_info)
            return schema_info_str
        except Exception as e:
            return f"Error retrieving database schema: {str(e)}"


def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query and return the results in a formatted string.

    Args:
        sql_query (str): The SQL query to execute

    Returns:
        str: Formatted string containing the query results or error message
    """
    try:
        # Create engine
        engine = sa.create_engine(connection)

        # Execute query and fetch results
        with engine.connect() as conn:
            result = conn.execute(sa.text(sql_query))

            # Get column names
            columns = result.keys()

            # Fetch all rows
            rows = result.fetchall()

            if not rows:
                return "Query executed successfully. No results returned."

            # Calculate column widths
            col_widths = {col: len(str(col)) for col in columns}
            for row in rows:
                for col, value in zip(columns, row):
                    col_widths[col] = max(col_widths[col], len(str(value)))

            # Format the output
            output = []

            # Create header
            header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
            output.append(header)

            # Add separator
            separator = "-" * len(header)
            output.append(separator)

            # Add rows
            for row in rows:
                formatted_row = " | ".join(
                    str(value).ljust(col_widths[col])
                    for col, value in zip(columns, row)
                )
                output.append(formatted_row)

            # Add result count
            output.append(f"\n{len(rows)} rows returned.")

            return "\n".join(output)

    except Exception as e:
        return f"Error executing query: {str(e)}"


# %%
tools = [execute_sql_query]
model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

from IPython.display import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

# %%
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class State(MessagesState):
    pass


SYSTEM_MESSAGE_PROMPT = """You are a database expert assistant who can interpret the user queries and provide a helpful response.

Use the provided toolset efficiently to generate helpful response for the user.
"""

sys_msg = SystemMessage(content=SYSTEM_MESSAGE_PROMPT)


# Node
def assistant(state: MessagesState):
    return {"messages": [model_with_tools.invoke([sys_msg] + state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

graph = builder.compile()
