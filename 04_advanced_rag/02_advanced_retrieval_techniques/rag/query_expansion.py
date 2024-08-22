import json
from langchain_openai import AzureChatOpenAI

from llm.chain import GeneralChain
from llm.prompt_templates import QueryExpansionTemplate, QuestionList
from config import settings


class QueryExpansion:
    @staticmethod
    def generate_response(query: str, to_expand_to_n: int) -> list[str]:
        query_expansion_template = QueryExpansionTemplate()
        prompt_template = query_expansion_template.create_template(to_expand_to_n)
        model = AzureChatOpenAI(azure_deployment=settings.AZURE_DEPLOYMENT_NAME, 
                                api_version=settings.AZURE_API_VERSION,
                                temperature=0)

        chain = GeneralChain().get_chain(llm=model, output_key="questions", template=prompt_template)

        response = chain.invoke({"question": query})
        question_obj = QuestionList(**json.loads(response["questions"]))
        
        stripped_queries = [
            stripped_item for item in question_obj.questions if (stripped_item := item.strip())
        ]

        return stripped_queries
    
