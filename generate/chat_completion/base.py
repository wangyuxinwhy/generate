from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar, Iterator, Type, TypeVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing_extensions import Self, Unpack

from generate.chat_completion.message import Prompt
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.http import HttpClient
from generate.model import GenerateModel, ModelParameters

O = TypeVar('O', bound=BaseModel)  # noqa: E741

if TYPE_CHECKING:
    from generate.modifiers.structure import Structure, StructureKwargs


logger = logging.getLogger(__name__)


class ChatCompletionModel(GenerateModel[Prompt, ChatCompletionOutput], ABC):
    model_task: ClassVar[str] = 'chat_completion'
    model_type: ClassVar[str]

    @abstractmethod
    def stream_generate(self, prompt: Prompt, **kwargs: Any) -> Iterator[ChatCompletionStreamOutput]:
        ...

    @abstractmethod
    def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        ...

    def structure(
        self, instruction: str, output_structure_type: Type[O], **kwargs: Unpack['StructureKwargs']
    ) -> Structure[Self, O]:
        from generate.modifiers.structure import Structure

        return Structure(
            self,
            instruction=instruction,
            output_structure_type=output_structure_type,
            **kwargs,
        )


class RemoteChatCompletionModel(ChatCompletionModel):
    def __init__(
        self,
        parameters: ModelParameters,
        settings: BaseSettings,
        http_client: HttpClient,
    ) -> None:
        self.parameters = parameters
        self.settings = settings
        self.http_client = http_client
