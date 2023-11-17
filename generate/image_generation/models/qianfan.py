from __future__ import annotations

import base64
from typing import Any, Literal, Optional

from pydantic import Field
from typing_extensions import Annotated, Self

from generate.http import HttpClient, HttpxPostKwargs
from generate.image_generation.base import GeneratedImage, ImageGenerationModel, ImageGenerationOutput
from generate.model import ModelParameters
from generate.platforms.baidu import QianfanSettings, QianfanTokenMixin

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


class QianfanImageGeneration(ImageGenerationModel[QianfanImageGenerationParameters], QianfanTokenMixin):
    model_type = 'qianfan'

    def __init__(
        self,
        model: str = 'sd_xl',
        settings: QianfanSettings | None = None,
        parameters: QianfanImageGenerationParameters | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or QianfanImageGenerationParameters()
        super().__init__(parameters)

        self.model = model
        self.settings = settings or QianfanSettings()  # type: ignore
        self.http_client = http_client or HttpClient()

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
                'access_token': self.token,
            },
        }

    def _image_generation(self, prompt: str, parameters: QianfanImageGenerationParameters) -> ImageGenerationOutput:
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, response.json())

    async def _async_image_generation(self, prompt: str, parameters: QianfanImageGenerationParameters) -> ImageGenerationOutput:
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        return self._construct_model_output(prompt, response.json())

    def _construct_model_output(self, prompt: str, response_data: dict[str, Any]) -> ImageGenerationOutput:
        images: list[GeneratedImage] = []
        for image_data in response_data['data']:
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
                'usage': response_data['usage'],
                'task_id': response_data['id'],
            },
        )

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
