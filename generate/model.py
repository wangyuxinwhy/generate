from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_serializer


class ModelInfo(BaseModel):
    task: Literal['chat_completion', 'image_generation', 'text_to_speech']
    type: str
    name: str

    @property
    def model_id(self) -> str:
        return f'{self.type}/{self.name}'


class ModelOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_info: ModelInfo
    cost: Optional[float] = None
    extra: Dict[str, Any] = {}

    @model_serializer(mode='wrap')
    def ser_model(self, handler) -> Any:  # noqa: ANN001
        output = handler(self)
        output.pop('debug')
        return output
