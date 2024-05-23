from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar, Iterator, List, Type, TypeVar, get_type_hints

from pydantic import BaseModel
from typing_extensions import Self, Unpack, override

from generate.chat_completion.message import Prompt
from generate.chat_completion.message.converter import MessageConverter
from generate.chat_completion.message.core import Messages
from generate.chat_completion.message.utils import ensure_messages
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.chat_completion.tool import SupportToolCall
from generate.http import HttpClient, HttpxPostKwargs
from generate.model import GenerateModel, ModelParameters
from generate.platforms import PlatformSettings
from generate.utils import sync_aiter

O = TypeVar('O', bound=BaseModel)  # noqa: E741

if TYPE_CHECKING:
    from generate.modifiers.agent import Agent, AgentKwargs
    from generate.modifiers.cache import CacheChatCompletionModel
    from generate.modifiers.hook import HookChatCompletionModel, HookModelKwargs
    from generate.modifiers.session import SessionChatCompletionModel
    from generate.modifiers.structure import StructureGenerateModel, StructureModelKwargs


logger = logging.getLogger(__name__)


class ChatCompletionModel(GenerateModel[Prompt, ChatCompletionOutput], ABC):
    model_task: ClassVar[str] = 'chat_completion'

    @abstractmethod
    def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        ...

    def stream_generate(self, prompt: Prompt, **kwargs: Any) -> Iterator[ChatCompletionStreamOutput]:
        return sync_aiter(self.async_stream_generate(prompt, **kwargs))

    def structure(
        self, output_structure_type: Type[O], **kwargs: Unpack['StructureModelKwargs']
    ) -> 'StructureGenerateModel[Self, O]':
        from generate.modifiers.structure import StructureGenerateModel

        return StructureGenerateModel(
            self,
            output_structure_type=output_structure_type,
            **kwargs,
        )

    def session(self) -> 'SessionChatCompletionModel':
        from generate.modifiers.session import SessionChatCompletionModel

        return SessionChatCompletionModel(model=self)

    def agent(self, **kwargs: Unpack['AgentKwargs']) -> 'Agent':
        from generate.modifiers.agent import Agent

        return Agent(model=self, **kwargs)

    def hook(self, **kwargs: Unpack['HookModelKwargs']) -> 'HookChatCompletionModel':
        from generate.modifiers.hook import HookChatCompletionModel

        return HookChatCompletionModel(model=self, **kwargs)

    def cache(self, cache_dir: Path | str | None = None) -> 'CacheChatCompletionModel':
        from generate.modifiers.cache import CacheChatCompletionModel

        return CacheChatCompletionModel(model=self, cache_dir=cache_dir)


class RemoteChatCompletionModel(ChatCompletionModel, ABC):
    settings: PlatformSettings
    http_client: HttpClient
    message_converter: MessageConverter
    available_models: ClassVar[List[str]] = []

    def __init__(
        self,
        model: str,
        parameters: ModelParameters,
        settings: PlatformSettings,
        http_client: HttpClient,
        message_converter: MessageConverter,
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.settings = settings
        self.http_client = http_client
        self.message_converter = message_converter

    @abstractmethod
    def _process_reponse(self, response: dict[str, Any]) -> ChatCompletionOutput:
        ...

    @abstractmethod
    def _process_stream_response(
        self, response: dict[str, Any], stream_manager: StreamManager
    ) -> ChatCompletionStreamOutput | None:
        ...

    @abstractmethod
    def _get_request_parameters(self, messages: Messages, stream: bool = False, **kwargs: Any) -> HttpxPostKwargs:
        ...

    def list_models(self) -> List[str]:
        return self.available_models

    def process_prompt(self, prompt: Prompt) -> Messages:
        messages = ensure_messages(prompt)
        if isinstance(self, SupportToolCall):
            self.process_messages_for_tool_call(messages)
        return messages

    @override
    def generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        timeout = kwargs.pop('timeout') if 'timeout' in kwargs else None
        messages = self.process_prompt(prompt)
        request_parameters = self._get_request_parameters(messages, **kwargs)
        request_parameters['timeout'] = timeout
        response = self.http_client.post(request_parameters=request_parameters)
        return self._process_reponse(response.json())

    @override
    async def async_generate(self, prompt: Prompt, **kwargs: Any) -> ChatCompletionOutput:
        timeout = kwargs.pop('timeout') if 'timeout' in kwargs else None
        messages = self.process_prompt(prompt)
        request_parameters = self._get_request_parameters(messages, **kwargs)
        request_parameters['timeout'] = timeout
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._process_reponse(response.json())

    @override
    def stream_generate(self, prompt: Prompt, **kwargs: Any) -> Iterator[ChatCompletionStreamOutput]:
        timeout = kwargs.pop('timeout') if 'timeout' in kwargs else None
        messages = self.process_prompt(prompt)
        request_parameters = self._get_request_parameters(messages, stream=True, **kwargs)
        request_parameters['timeout'] = timeout
        stream_manager = StreamManager(info=self.model_info)
        for line in self.http_client.stream_post(request_parameters=request_parameters):
            try:
                response = json.loads(line)
                if (output := self._process_stream_response(response, stream_manager)) and output:
                    yield output
            except json.JSONDecodeError:
                continue

    @override
    async def async_stream_generate(self, prompt: Prompt, **kwargs: Any) -> AsyncIterator[ChatCompletionStreamOutput]:
        timeout = kwargs.pop('timeout') if 'timeout' in kwargs else None
        messages = self.process_prompt(prompt)
        request_parameters = self._get_request_parameters(messages, stream=True, **kwargs)
        request_parameters['timeout'] = timeout
        stream_manager = StreamManager(info=self.model_info)
        async for line in self.http_client.async_stream_post(request_parameters=request_parameters):
            try:
                response = json.loads(line)
                if (output := self._process_stream_response(response, stream_manager)) and output:
                    yield output
            except json.JSONDecodeError:
                continue

    @classmethod
    def how_to_settings(cls) -> str:
        return f'{cls.__name__} Settings\n\n' + get_type_hints(cls)['settings'].how_to_settings()

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(model=name)  # type: ignore
