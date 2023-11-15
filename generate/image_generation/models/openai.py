from __future__ import annotations

import os
from typing import Any, Literal, Optional

from httpx import Response
from typing_extensions import Self, Unpack

from generate.http import HttpClient, HttpClientInitKwargs, HttpxPostKwargs
from generate.image_generation.base import ImageGenerationModel
from generate.image_generation.model_output import ImageGenerationOutput
from generate.parameters import ModelParameters


class OpenAIImageGenerationParameters(ModelParameters):
    quality: Optional[Literal['hd', 'standard']] = None
    response_format: Optional[Literal['url', 'b64_json']] = None
    size: Optional[Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']] = None
    style: Optional[Literal['vivid', 'natural']] = None
    user: Optional[str] = None


class OpenAIImageGeneration(ImageGenerationModel[OpenAIImageGenerationParameters]):
    model_type = 'openai'
    default_api_base = 'https://api.openai.com/v1/images/generations'

    def __init__(
        self,
        model: str = 'dall-e-3',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: OpenAIImageGenerationParameters | None = None,
        **kwargs: Unpack[HttpClientInitKwargs],
    ) -> None:
        parameters = parameters or OpenAIImageGenerationParameters()
        super().__init__(parameters)
        self.model = model
        self.api_base = api_base or self.default_api_base
        self.api_key = api_key or os.environ['OPENAI_API_KEY']
        self.http_client = HttpClient(**kwargs)

    def _get_request_parameters(self, text: str, parameters: OpenAIImageGenerationParameters) -> HttpxPostKwargs:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        json_data = {
            'model': self.model,
            'prompt': text,
            **parameters.model_dump(exclude_none=True),
        }
        return {
            'url': self.api_base,
            'json': json_data,
            'headers': headers,
        }

    def _image_generation(self, prompt: str, parameters: OpenAIImageGenerationParameters) -> ImageGenerationOutput:
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, parameters, response)

    def _construct_model_output(
        self, text: str, parameters: OpenAIImageGenerationParameters, response: Response
    ) -> ImageGenerationOutput:
        output = ImageGenerationOutput(model_info=self.model_info)
        output.extra['response'] = response
        return output

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
