import json
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from config import settings
from pydantic import BaseModel

from llm.prompt_templates import BasePromptTemplate

"""
At the time of exploration, Azure has not yet integrated the Structured Output feature in its API.
Hence, we work our way through prompting and response_format as {"type": "json_object"} to enable JSON Mode.

References:
https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode?tabs=python
"""

class Question(BaseModel):
    questions: list[str]

class QueryExpansionTemplate(BasePromptTemplate):
    prompt: str = """
    You are an expert in understanding user questions to extract relevant and accurate information from the training manuals. 
    Your task is to generate {to_expand_to_n} different versions of the given user question 
    to retrieve relevant documents from a vector database. Always ensure that the generated questions shall be utilized to retrieve
    relevant training manual related content stored in the vector database using distance-based similarity search. By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions as a json list.
    Example JSON Schema:
    {json_schema_example}
    Original question: {question}"""

    @property
    def json_schema_example(self) -> str:
        return Question.model_json_schema()

    def create_template(self, to_expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "json_schema_example": self.json_schema_example,
                "to_expand_to_n": to_expand_to_n,
            },
        )

# To run as standalone script, CD to project root dir which is ./02_advanced_retrieval_techniques and run
# python -m examples.azure_openai_structured_output_example
if __name__ == "__main__":
    query_expansion_template = QueryExpansionTemplate()
    prompt_template = query_expansion_template.create_template(to_expand_to_n=5)
    

    response_format = {"type": "json_object"}

    model = AzureChatOpenAI(azure_deployment=settings.AZURE_DEPLOYMENT_NAME, 
                                api_version=settings.AZURE_API_VERSION,
                                response_format=response_format,
                                temperature=0)
    
    structured_output_chain = prompt_template | model
    example_json_format = {"questions": ["question 1", "question 2", "question 3"]}
    response = structured_output_chain.invoke({"question": "how did the world evolve ?"})
    print("**** RESPONSE ****")
    print(response)
    question_obj = Question(**json.loads(response.content))
    print("**** QUESTION OBJECT ****")
    print(question_obj)
    print("**** QUESTIONS ****")
    print(question_obj.questions)