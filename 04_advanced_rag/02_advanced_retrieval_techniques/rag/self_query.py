from langchain_openai import AzureChatOpenAI
from llm.chain import GeneralChain
from llm.prompt_templates import SelfQueryTemplate
from config import settings


class SelfQuery:
    @staticmethod
    def generate_response(query: str) -> str | None:
        prompt = SelfQueryTemplate().create_template()
        model = AzureChatOpenAI(azure_deployment=settings.AZURE_DEPLOYMENT_NAME, 
                                api_version=settings.AZURE_API_VERSION,
                                temperature=0)

        chain = GeneralChain().get_chain(
            llm=model, output_key="metadata_filter_value", template=prompt
        )

        response = chain.invoke({"question": query})
        result = response.get("metadata_filter_value", "none")
        
        if result.lower() == "none":
            return None

        return result
