from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from typing_extensions import Self

from generate.parameters import ModelParameters
from generate.text_to_speech.model_output import TextToSpeechModelOutput

P = TypeVar('P', bound=ModelParameters)


class TextToSpeechModel(Generic[P], ABC):
    model_type: ClassVar[str]

    def __init__(self, parameters: P) -> None:
        self.parameters = parameters

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        ...

    @abstractmethod
    def _generate(self, text: str, parameters: P) -> TextToSpeechModelOutput:
        ...

    @abstractmethod
    async def _async_generate(self, text: str, parameters: P) -> TextToSpeechModelOutput:
        ...

    @property
    def model_id(self) -> str:
        return f'{self.model_type}/{self.name}'

    def generate(self, text: str, **override_parameters: Any) -> TextToSpeechModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        model_output = self._generate(text, parameters)
        model_output.debug['input_text'] = text
        model_output.debug['parameters'] = parameters
        return model_output

    async def async_generate(self, text: str, **override_parameters: Any) -> TextToSpeechModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        model_output = await self._async_generate(text, parameters)
        model_output.debug['input_text'] = text
        model_output.debug['parameters'] = parameters
        return model_output

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )
