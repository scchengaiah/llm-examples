from langchain_community.llms.openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import format_document, PromptTemplate

from src.langchain_helper.model_config import LLMModel, ModelConfig
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from src.langchain_helper.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

from operator import itemgetter

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.model_type = config.model_type
        self.secrets = config.secrets
        self.callback_handler = config.callback_handler
        account_tag = self.secrets["CF_ACCOUNT_TAG"]
        self.gateway_url = (
            f"https://gateway.ai.cloudflare.com/v1/{account_tag}/k-1-gpt/openai"
        )
        self.setup()

    def setup(self):
        if self.model_type == LLMModel.GPT_3_5:
            self.setup_gpt()
        elif self.model_type == LLMModel.CLAUDE:
            self.setup_claude()

    def setup_gpt(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0125",
            temperature=0.2,
            api_key=self.secrets["OPENAI_API_KEY"],
            max_tokens=1000,
            callbacks=[self.callback_handler],
            streaming=True,
            base_url=self.gateway_url,
        )

    def setup_claude(self):
        self.llm = ChatOpenAI(
            model_name="mixtral-8x7b-32768",
            temperature=0.2,
            api_key=self.secrets["GROQ_API_KEY"],
            max_tokens=3000,
            callbacks=[self.callback_handler],
            streaming=True,
            base_url="https://api.groq.com/openai/v1",
        )

    def get_chain(self, vectorstore):
        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)

        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | OpenAI()
            | StrOutputParser(),
        )
        _context = {
            "context": itemgetter("standalone_question")
            | vectorstore.as_retriever()
            | _combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        conversational_qa_chain = _inputs | _context | QA_PROMPT | self.llm

        return conversational_qa_chain