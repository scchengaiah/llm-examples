from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_openai import AzureChatOpenAI
import os

from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field

from langchain.globals import set_debug
set_debug(True)

from dotenv import load_dotenv

result =load_dotenv(dotenv_path="D:/gitlab/learnings/artificial-intelligence/llm-examples/01_guardrails/00_nemo_guardrails_exploration/.env")

print(f"Env loaded: {result}")


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

define user ask math question
  "What is the square root of 7?"
  "What is the formula for the area of a circle?"

define flow
  user ask math question
  $result = execute Calculator(question=$user_message)
  bot respond    

define flow
    user ...
    $answer = execute CompanyPolicyExpert(question=$last_user_message)
    bot $answer
"""

yaml_content = """
models:
- type: main
  engine: azure
"""

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_LLM_MODEL_DEPLOYMENT_NAME")
)
config =  RailsConfig.from_content(
    yaml_content=yaml_content,
    colang_content=colang_content
)

from langchain.chains import LLMMathChain

tools = []

class CalculatorInput(BaseModel):
    question: str = Field()


llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools.append(
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput,
    )
)

rails = RunnableRails(
                        config, llm=llm,
                        tools=tools,
    input_key="question",
                    )

prompt = ChatPromptTemplate.from_template("{question}")
chain = RunnablePassthrough() | prompt | (rails | llm)

print(chain.invoke({"question": "What is 5+5*5/5?"}))
