from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, Self, TypedDict, Unpack, override

from generate.http import HttpClient, HttpxPostKwargs
from generate.model import ModelParameters
from generate.platforms.openai import OpenAISettings
from generate.text_to_speech.base import RemoteTextToSpeechModel, TextToSpeechOutput


class OpenAISpeechParameters(ModelParameters):
    voice: str = 'alloy'
    response_format: Optional[Literal['mp3', 'aac', 'opus', 'flac']] = None
    speed: Annotated[Optional[float], Field(ge=0.25, le=4.0)] = None


class OpenAISpeechParametersDict(TypedDict, total=False):
    voice: str
    response_format: Optional[Literal['mp3', 'aac', 'opus', 'flac']]
    speed: Optional[float]


class OpenAISpeech(RemoteTextToSpeechModel):
    model_type = 'openai'

    parameters: OpenAISpeechParameters
    settings: OpenAISettings

    def __init__(
        self,
        model: str = 'tts-1',
        settings: OpenAISettings | None = None,
        parameters: OpenAISpeechParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or OpenAISpeechParameters()
        settings = settings or OpenAISettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model

    def _get_request_parameters(self, text: str, parameters: OpenAISpeechParameters) -> HttpxPostKwargs:
        json_data = {
            'model': self.model,
            'input': text,
            **parameters.custom_model_dump(),
        }
        headers = {
            'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.settings.api_base + 'audio/speech',
            'json': json_data,
            'headers': headers,
        }

    @override
    def generate(self, prompt: str, **kwargs: Unpack[OpenAISpeechParametersDict]) -> TextToSpeechOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return TextToSpeechOutput(
            model_info=self.model_info,
            audio=response.content,
            audio_format=parameters.response_format or 'mp3',
            cost=self.calculate_cost(prompt),
        )

    @override
    async def async_generate(self, prompt: str, **kwargs: Unpack[OpenAISpeechParametersDict]) -> TextToSpeechOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return TextToSpeechOutput(
            model_info=self.model_info,
            audio=response.content,
            audio_format=parameters.response_format or 'mp3',
            cost=self.calculate_cost(prompt),
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)

    def calculate_cost(self, text: str) -> float | None:
        dollar_to_yuan = 7
        if self.model == 'tts-1':
            return (len(text) / 1000) * (0.015 * dollar_to_yuan)

        if self.model == 'tts-1-hd':
            return (len(text) / 1000) * (0.03 * dollar_to_yuan)

        return None
