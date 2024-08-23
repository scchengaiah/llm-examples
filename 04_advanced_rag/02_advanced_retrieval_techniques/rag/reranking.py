import json
from langchain_openai import AzureChatOpenAI

from llm.chain import GeneralChain
from llm.prompt_templates import RerankingTemplate, PassageList
from config import settings


class Reranker:
    @staticmethod
    def generate_response(
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        reranking_template = RerankingTemplate()
        prompt_template = reranking_template.create_template(keep_top_k=keep_top_k)

        model = AzureChatOpenAI(azure_deployment=settings.AZURE_DEPLOYMENT_NAME, 
                                api_version=settings.AZURE_API_VERSION,
                                temperature=0)
        chain = GeneralChain().get_chain(
            llm=model, output_key="reranked_passages", template=prompt_template
        )

        stripped_passages = [
            f"<Passage>\n{stripped_item}\n</Passage>" for item in passages if (stripped_item := item.strip())
        ]
        passages = "\n".join(stripped_passages)
        response = chain.invoke({"question": query, "passages": passages})

        passage_list_obj = PassageList(**json.loads(response["reranked_passages"]))
        
        
        stripped_passages = [
            stripped_item
            for item in passage_list_obj.passage_list
            if (stripped_item := item.passage.strip())
        ]

        return stripped_passages
