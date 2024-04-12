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

PROMPT = PromptTemplate.from_template(template="You are a friendly chatbot assistant who can answer my question.\n{question}")





class CustomChain:
    def __init__(self, yaml_content, colang_content, llm_for_rails):
        config = RailsConfig.from_content(
            yaml_content=yaml_content,
            colang_content=colang_content
        )
        self.rails = LLMRails(config, llm=llm_for_rails)
        self.rails.register_action(self.call_llm_chain)

    async def call_llm_chain(self, user_query: str, chat_history: [], llm: Optional[BaseLLM], ) -> str:
        call_config = RunnableConfig(callbacks=[streaming_handler_var.get()])
        loaded_chat_history = loads(chat_history)
        llm.callbacks = [streaming_handler_var.get()]
        chain = PROMPT | llm

        # response = await chain.ainvoke(user_query, config=call_config)
        response = await chain.ainvoke(user_query)
        # print("*" * 25)
        # print("RESPONSE:")
        # print(response)
        # print("*" * 25)
        return response.content


            # Or do something else with the token



        # print(rails.explain().colang_history)

    async def process_tokens(self, streaming_handler_obj):
        async for chunk in streaming_handler_obj:
            print(f"{chunk}", end="", flush=True)
    async def stream_query(self, input, streaming_handler_obj):
        asyncio.create_task(self.process_tokens(streaming_handler_obj))

        messages = [
            {"role": "context", "content": {"chat_history": dumps(input["chat_history"])}},
            {"role": "user", "content": input["question"]}
        ]

        return await self.rails.generate_async(messages=messages,
                                          streaming_handler=streaming_handler_obj)