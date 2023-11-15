from __future__ import annotations

import os
from typing import Any, ClassVar, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, Self, Unpack, override

from generate.http import HttpClient, HttpClientInitKwargs, HttpxPostKwargs
from generate.parameters import ModelParameters
from generate.text_to_speech.base import TextToSpeechModel
from generate.text_to_speech.model_output import TextToSpeechOutput


class OpenAISpeechParameters(ModelParameters):
    voice_id: Annotated[str, Field(serialization_alias='voice')] = 'alloy'
    response_format: Optional[Literal['mp3', 'aac', 'opus', 'flac']] = None
    speed: Annotated[Optional[float], Field(ge=0.25, le=4.0)] = None


class OpenAISpeech(TextToSpeechModel[OpenAISpeechParameters]):
    model_type = 'openai'
    default_api_base: ClassVar[str] = 'https://api.openai.com/v1/audio/speech'

    def __init__(
        self,
        model: str = 'tts-1',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: OpenAISpeechParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or OpenAISpeechParameters()
        super().__init__(parameters)
        self.model = model
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or self.default_api_base
        self.api_key = api_key or os.environ['OPENAI_API_KEY']
        self.http_client = HttpClient(**kwargs)

    def _get_request_parameters(self, text: str, parameters: OpenAISpeechParameters) -> HttpxPostKwargs:
        parameters_dict = parameters.model_dump(exclude_none=True, by_alias=True)
        json_data = {
            'model': self.model,
            'input': text,
            **parameters_dict,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.api_base,
            'json': json_data,
            'headers': headers,
        }

    def _text_to_speech(self, text: str, parameters: OpenAISpeechParameters) -> TextToSpeechOutput:
        request_parameters = self._get_request_parameters(text, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return TextToSpeechOutput(
            model_info=self.model_info,
            audio=response.content,
            audio_format=parameters.response_format or 'mp3',
            cost=self.calculate_cost(text),
        )

    async def _async_text_to_speech(self, text: str, parameters: OpenAISpeechParameters) -> TextToSpeechOutput:
        request_parameters = self._get_request_parameters(text, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return TextToSpeechOutput(
            model_info=self.model_info,
            audio=response.content,
            audio_format=parameters.response_format or 'mp3',
            cost=self.calculate_cost(text),
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)

    def calculate_cost(self, text: str) -> float | None:
        dollar_to_yuan = 7
        if self.model == 'tts-1':
            return (len(text) / 1000) * (0.015 * dollar_to_yuan)

        if self.model == 'tts-1-hd':
            return (len(text) / 1000) * (0.03 * dollar_to_yuan)

        return None
