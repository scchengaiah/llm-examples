from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from pydantic import BaseModel



class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass


class QueryExpansionTemplate_OLD(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to generate {to_expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, to_expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "to_expand_to_n": to_expand_to_n,
            },
        )

class QuestionList(BaseModel):
    questions: list[str]

class QueryExpansionTemplate(BasePromptTemplate):
    prompt: str = """
    You are an expert in understanding user questions to extract relevant and accurate information from the training manuals. 
    Your task is to generate {to_expand_to_n} different versions of the given user question 
    to retrieve relevant documents from a vector database. Keep in mind that the context of the generated questions should be relevant to the Original question.
    Always ensure that the generated questions shall be utilized to retrieve
    relevant training manual related content stored in the vector database using distance-based similarity search. By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions as a json list. Output only the raw json and do not include any markdown formattting.
    Example JSON Schema:
    {json_schema_example}
    Original question: {question}"""

    @property
    def json_schema_example(self) -> str:
        return QuestionList.model_json_schema()

    def create_template(self, to_expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "json_schema_example": self.json_schema_example,
                "to_expand_to_n": to_expand_to_n,
            },
        )


class SelfQueryTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user or author id. 
    Your response should consists of only the extracted id (e.g. 1345256), nothing else.
    If you cannot find the author id, return the string "None".
    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])

class Passage(BaseModel):
    passage: str
    rank:int

class PassageList(BaseModel):
    passage_list: list[Passage]

class RerankingTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to rerank passages related to a query
    based on their relevance. 
    The most relevant passages should be identified with high accuracy. 
    You should only pick at max {keep_top_k} passages.
    The provided and reranked passages should be formatted in json as per the provided json schema. Output only the raw json and do not include any markdown formattting'.
    Example JSON Schema:
    {json_schema_example}

    The following are passages related to this query: {question}.
    
    Passages: 
    {passages}
    """

    def create_template(self, keep_top_k: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question", "passages"],
            partial_variables={"keep_top_k": keep_top_k, "json_schema_example": self.json_schema_example,},
        )

    @property
    def json_schema_example(self) -> str:
        return PassageList.model_json_schema()
