from __future__ import annotations

import base64
from typing import Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, Self, Unpack, override

from generate.http import HttpClient, HttpxPostKwargs, ResponseValue
from generate.image_generation.base import GeneratedImage, ImageGenerationOutput, RemoteImageGenerationModel
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms.baidu import QianfanSettings, QianfanTokenManager

ValidSize = Literal[
    '768x768',
    '768x1024',
    '1024x768',
    '576x1024',
    '1024x576',
    '1024x1024',
]


class QianfanImageGenerationParameters(ModelParameters):
    size: Optional[ValidSize] = None
    n: Optional[Annotated[int, Field(ge=1, le=4)]] = None
    negative_prompt: Optional[str] = None
    steps: Optional[Annotated[int, Field(ge=1, le=50)]] = None
    sampler: Optional[str] = Field(default=None, serialization_alias='sampler_index')
    user: Optional[str] = Field(default=None, serialization_alias='user_id')


class QianfanImageGenerationParametersDict(ModelParametersDict, total=False):
    size: Optional[ValidSize]
    n: Optional[int]
    negative_prompt: Optional[str]
    steps: Optional[int]
    sampler: Optional[str]
    user: Optional[str]


class QianfanImageGeneration(RemoteImageGenerationModel):
    model_type = 'qianfan'

    parameters: QianfanImageGenerationParameters
    settings: QianfanSettings

    def __init__(
        self,
        model: str = 'sd_xl',
        parameters: QianfanImageGenerationParameters | None = None,
        settings: QianfanSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or QianfanImageGenerationParameters()
        settings = settings or QianfanSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.model = model
        self.token_manager = QianfanTokenManager(self.settings, self.http_client)

    @override
    def generate(self, prompt: str, **kwargs: Unpack[QianfanImageGenerationParametersDict]) -> ImageGenerationOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, response.json())

    @override
    async def async_generate(
        self, prompt: str, **kwargs: Unpack[QianfanImageGenerationParametersDict]
    ) -> ImageGenerationOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, response.json())

    def _get_request_parameters(self, prompt: str, parameters: QianfanImageGenerationParameters) -> HttpxPostKwargs:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        json_data = {
            'prompt': prompt,
            **parameters.custom_model_dump(),
        }
        return {
            'url': self.settings.image_generation_api_base + self.model,
            'json': json_data,
            'headers': headers,
            'params': {
                'access_token': self.token_manager.token,
            },
        }

    def _construct_model_output(self, prompt: str, response_value: ResponseValue) -> ImageGenerationOutput:
        images: list[GeneratedImage] = []
        for image_data in response_value['data']:
            image = GeneratedImage(
                prompt=prompt,
                image_format='png',
                content=base64.b64decode(image_data['b64_image']),
            )
            images.append(image)
        return ImageGenerationOutput(
            model_info=self.model_info,
            images=images,
            extra={
                'usage': response_value['usage'],
                'task_id': response_value['id'],
            },
        )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        return cls(model=name)
