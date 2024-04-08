import os

from langchain_community.embeddings import BedrockEmbeddings


class AWSBedrockEmbeddings:

    def __init__(self):
        self._embeddings = None
        self._validate_aws_env_variables()
        self._region_name = os.environ["AWS_REGION"]
        self._model_id = os.environ["AWS_LLM_EMBEDDINGS_ID"]
        self.initialize_embeddings()

    def initialize_embeddings(self):
        self._embeddings = BedrockEmbeddings(region_name=self._region_name, model_id=self._model_id)

    @property
    def region_name(self):
        return self._region_name

    @property
    def model_id(self):
        return self._model_id

    @property
    def embeddings(self):
        return self._embeddings

    def _validate_aws_env_variables(self):
        if "AWS_REGION" not in os.environ:
            raise ValueError("AWS_REGION environment variable not set")
        if "AWS_LLM_EMBEDDINGS_ID" not in os.environ:
            raise ValueError("AWS_LLM_EMBEDDINGS_ID environment variable not set")
        if "AWS_ACCESS_KEY_ID" not in os.environ:
            raise ValueError("AWS_ACCESS_KEY_ID environment variable not set")
        if "AWS_SECRET_ACCESS_KEY" not in os.environ:
            raise ValueError("AWS_SECRET_ACCESS_KEY environment variable not set")