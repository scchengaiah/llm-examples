from enum import Enum
from typing import Dict, Optional, Callable, Any

from pydantic import BaseModel
from pydantic.v1 import validator


class LLMModel(Enum):
    GPT_3_5 = "GPT-3.5"
    CLAUDE = "Claude"


class ModelConfig(BaseModel):
    model_type: LLMModel
    secrets: Dict[str, Any]
    callback_handler: Optional[Callable] = None

    @validator("model_type", pre=True, always=True)
    def validate_model_type(cls, v):
        if v not in LLMModel.__members__().values():
            raise ValueError(f"Unsupported model type: {v}")
        return v
