import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import BaseModel

from llm.prompt_templates import BasePromptTemplate

"""
References:
https://platform.openai.com/docs/guides/structured-outputs/structured-outputs
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
    Provide these alternative questions as a list.
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

# To run as standalone script, CD to project root dir which is ./02_advanced_retrieval_techniques and run
# python -m examples.openai_structured_output_example
if __name__ == "__main__":
    query_expansion_template = QueryExpansionTemplate()
    prompt_template = query_expansion_template.create_template(to_expand_to_n=5)
    

    response_format = type_to_response_format_param(Question)

    model = ChatOpenAI(model="gpt-4o-mini", 
                                response_format=response_format, # We can also directly pass Question as value to this arg.
                                temperature=0)
    
    structured_output_chain = prompt_template | model
    response = structured_output_chain.invoke({"question": "how did the world evolve?"})
    print("**** RESPONSE ****")
    print(response)
    question_obj = Question(**json.loads(response.content))
    print("**** QUESTION OBJECT ****")
    print(question_obj)
    print("**** QUESTIONS ****")
    print(question_obj.questions)