import sys

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_openai import AzureChatOpenAI
import os

from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field

from langchain.globals import set_debug
set_debug(False)

from dotenv import load_dotenv
from typing import Optional
from langchain_core.runnables import Runnable
from langchain_core.load.load import loads
from nemoguardrails.actions import action
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import RunnableConfig
from nemoguardrails import LLMRails, RailsConfig
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.load.dump import dumps
from nemoguardrails.context import streaming_handler_var
from nemoguardrails.streaming import StreamingHandler
import asyncio
from langchain_core.prompts import  PromptTemplate
from src.custom.impl.test_guardrails_streaming_class import CustomChain

result =load_dotenv(dotenv_path="D:/gitlab/learnings/artificial-intelligence/llm-examples/01_guardrails/00_nemo_guardrails_exploration/.env")

print(f"Env loaded: {result}")

PROMPT = PromptTemplate.from_template(template="You are a friendly chatbot assistant who can answer my question.\n{question}")



colang_content = """
define user express greeting
  "hello"
  "hi"

define bot express greeting
  "Hello there!! How can I help you today on our company policies?"
  "Hi there!! How can I help you today on our company policies?"

define flow hello
    user express greeting
    bot express greeting


define user enquires well-being
    "How are you ?"
    "How is your health ?"
    "Are you ok ?"
    "How are you feeling ?"

define bot responds well-being
    "As a chatbot, I do not have any feelings or emotions. However, I would be happy to assist you with any queries on our company policies."

define flow well-being
    user enquires well-being
    bot responds well-being

define user asks capabilities
    "How can you help me ?"
    "what are your capabilities ?"
    "what is your expertise ?"

define bot responds capabilities
    "I can answer questions related to our company policies. If you have some questions about company policies, feel free to ask."

define flow capabilities
    user asks capabilities
    bot responds capabilities

define user express gratitude
  "thank you"
  "thanks"

define bot respond gratitude
  "You're welcome. If you have any other question, feel free to ask me."

define flow gratitude
    user express gratitude
    bot respond gratitude

define user express appreciation
    "well done"
    "Good job"

define bot respond appreciation
    "Thank you. If you have any other question, feel free to ask me."

define flow appreciation
    user express appreciation
    bot respond appreciation

define user express insult
  "You are stupid"

define flow
  user express insult
  bot express calmly willingness to help

define user ask about Albert Einstein
  Explain about Albert Einstein
  

define flow ask about Albert Einstein
    user ask about Albert Einstein
    $answer = execute call_llm_chain(user_query=$user_message, chat_history=$chat_history)
    bot $answer
"""

yaml_content = """
models:
- type: main
  engine: azure

"""


chain_without_llm_configured = PROMPT

chat_history = [
        HumanMessage(content="Explain our company's leave policy ?"),
        AIMessage(
            content="Employees are eligible for 30 days of regular leaves for a given calendar year (1st Jan till 31st Dec) and must apply for planned leaves with prior approval from their project manager and designated reporting manager. In case of emergency, employees must inform their immediate superior and HR, and all leaves must be applied through the Intelizign Intranet Portal."),
        HumanMessage(content="Explain our company loan policy"),
        AIMessage(
            content="Our loan policy allows relocated employees in Germany to request a loan for a flat deposit up to 3,000€. To request a loan, employees must email the HR department with the purpose and required amount, and sign a document prepared by HR before repayment within one financial year.")
    ]

llm_for_rails = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_LLM_MODEL_DEPLOYMENT_NAME"),
    temperature=0,
    max_tokens=3000,
    streaming=True
)

query = "Explain about albert Einstein in 150 words ?"


async def execute_rails_wrapper():
    await execute_rails()


async def execute_rails():
    streaming_handler_obj = StreamingHandler()
    streaming_handler_var.set(streaming_handler_obj)
    custom_chain_obj = CustomChain(yaml_content, colang_content, llm_for_rails)
    result = await custom_chain_obj.stream_query({"question": query, "chat_history": chat_history},
                                        streaming_handler_obj)

    print("*" * 50)
    print(result)


"""

@action(is_system_action=True)
async def call_llm(user_query: str,llm: Optional[BaseLLM]) -> str:
    call_config = RunnableConfig(callbacks=[streaming_handler_var.get()])
    response = await llm.ainvoke(user_query, config=call_config)
    return response.content

# @action(is_system_action=True)
async def call_llm_chain(user_query: str,  chat_history: [], llm: Optional[BaseLLM],) -> str:
    call_config = RunnableConfig(callbacks=[streaming_handler_var.get()])
    loaded_chat_history = loads(chat_history)
    llm.callbacks = [streaming_handler_var.get()]
    chain = chain_without_llm_configured | llm

    #response = await chain.ainvoke(user_query, config=call_config)
    response = await chain.ainvoke(user_query)
    #print("*" * 25)
    #print("RESPONSE:")
    #print(response)
    #print("*" * 25)
    return response.content

config = RailsConfig.from_content(
  	yaml_content=yaml_content,
    colang_content=colang_content
)
# We go with Azure OpenAI LLM considering the optimization of prompts with Bedrock.
rails = LLMRails(config, llm=llm_for_rails)
rails.register_action(call_llm_chain)

streaming_handler = StreamingHandler()
streaming_handler_var.set(streaming_handler)

async def process_tokens(streaming_handler_obj):
    async for chunk in streaming_handler_obj:
        print(f"{chunk}", end="", flush=True)
        # Or do something else with the token



async def demo_streaming_from_custom_action():
    asyncio.create_task(process_tokens(streaming_handler))

    query = "Explain about albert Einstein in 150 words ?"

    chat_history = [
        HumanMessage(content="Explain our company's leave policy ?"),
        AIMessage(
            content="Employees are eligible for 30 days of regular leaves for a given calendar year (1st Jan till 31st Dec) and must apply for planned leaves with prior approval from their project manager and designated reporting manager. In case of emergency, employees must inform their immediate superior and HR, and all leaves must be applied through the Intelizign Intranet Portal."),
        HumanMessage(content="Explain our company loan policy"),
        AIMessage(
            content="Our loan policy allows relocated employees in Germany to request a loan for a flat deposit up to 3,000€. To request a loan, employees must email the HR department with the purpose and required amount, and sign a document prepared by HR before repayment within one financial year.")
    ]

    #chat_history = []

    messages = [{"role": "context", "content": {"chat_history": dumps(chat_history)}},
                {"role": "user", "content": query}]

    #messages = [{"role": "user", "content": query}]

    result = await stream_query({"question": query, "chat_history": chat_history}, streaming_handler)
    print("*" * 50)
    print(result)

    #print(rails.explain().colang_history)

async def stream_query(input, streaming_handler_obj):
    messages = [
        {"role": "context", "content": {"chat_history": dumps(input["chat_history"])}},
        {"role": "user", "content": input["question"]}
    ]

    return await rails.generate_async(messages=messages,
                                            streaming_handler=streaming_handler_obj)
"""
if __name__ == "__main__":
    asyncio.run(execute_rails_wrapper())