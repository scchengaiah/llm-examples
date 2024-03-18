from typing import Dict, Optional, Callable, Any

from pydantic import BaseModel

from src.init_llm_helper import LLMModel

class ModelConfig(BaseModel):
    llm_model_type: LLMModel
    secrets: Dict[str, Any]
    callback_handler: Optional[Callable] = None

