from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from src.langchain_helper.model_config import ModelConfig
from src.init_llm_helper import LLMModel
from src.langchain_helper.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.model_type = config.llm_model_type
        self.secrets = config.secrets
        self.callback_handler = config.callback_handler
        self.setup()

    def setup(self):
        if self.model_type == LLMModel.GPT_3_5:
            self.setup_gpt()
        elif self.model_type == LLMModel.CLAUDE:
            self.setup_claude()

    def setup_gpt(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0125",
            temperature=0,
            api_key=self.secrets["OPENAI_API_KEY"],
            max_tokens=1000,
            streaming=True
        )

    def setup_claude(self):
        self.llm = BedrockChat(
            region_name=self.secrets["AWS_REGION"],
            model_id=self.secrets["AWS_LLM_ID"],
            model_kwargs={"temperature":0, "max_tokens":1000},
            streaming=True
        )

    def get_chain(self, vectorstore):
        # https://python.langchain.com/docs/expression_language/how_to/routing
        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)

        def determine_chain(invocation_input):
            # Output of the _inputs execution is the standalone question in the format:
            # {"standalone_question": "question"}
            _inputs = RunnableParallel(
                question=RunnablePassthrough.assign(
                    chat_history=lambda x: get_buffer_string(x["chat_history"])
                )
                                    | CONDENSE_QUESTION_PROMPT
                                    | self.llm
                                    | StrOutputParser(),
            )
            # Takes the standalone question as the input and the context as the vectorstore.
            _context = {
                "context": itemgetter("question") | vectorstore.as_retriever() | _combine_documents,
                "question": lambda x: x["question"],
            }
            # Standalone question shall be generated if chat history is provided.
            if len(invocation_input["chat_history"]) > 0:
                return _inputs | _context | QA_PROMPT | self.llm
            else:
                return _context | QA_PROMPT | self.llm

        conversational_qa_chain = RunnablePassthrough() | RunnableLambda(determine_chain)
        return conversational_qa_chain
