from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Optional, Union

from pydantic import Base64Str, Field, HttpUrl
from typing_extensions import Annotated, Self, Unpack, override

from generate.http import HttpClient, HttpxPostKwargs, UnexpectedResponseError
from generate.image_generation.base import (
    GeneratedImage,
    ImageGenerationOutput,
    RemoteImageGenerationModel,
)
from generate.model import ModelParameters, ModelParametersDict
from generate.platforms.baidu import BaiduCreationSettings, BaiduCreationTokenManager

ValidSize = Literal[
    '512x512',
    '640x360',
    '360x640',
    '1024x1024',
    '1280x720',
    '720x1280',
    '2048x2048',
    '2560x1440',
    '1440x2560',
]


class BaiduImageGenerationParameters(ModelParameters):
    size: ValidSize = '1024x1024'
    n: Optional[Annotated[int, Field(ge=1, le=8)]] = None
    reference_image: Union[HttpUrl, Base64Str, None] = None
    change_degree: Optional[Annotated[int, Field(ge=1, le=10)]] = None

    def custom_model_dump(self) -> dict[str, Any]:
        output_data = {}
        width, height = self.size.split('x')
        output_data['width'] = int(width)
        output_data['height'] = int(height)
        n = self.n or 1
        output_data['image_num'] = n
        if self.reference_image:
            if isinstance(self.reference_image, HttpUrl):
                output_data['url'] = self.reference_image
            else:
                output_data['image'] = self.reference_image
        if self.change_degree:
            output_data['change_degree'] = self.change_degree
        return output_data


class BaiduImageGenerationParametersDict(ModelParametersDict, total=False):
    size: ValidSize
    n: Optional[int]
    reference_image: Union[HttpUrl, Base64Str, None]
    change_degree: Optional[int]


class BaiduImageGeneration(RemoteImageGenerationModel):
    model_type = 'baidu'

    parameters: BaiduImageGenerationParameters
    settings: BaiduCreationSettings

    def __init__(
        self,
        parameters: BaiduImageGenerationParameters | None = None,
        settings: BaiduCreationSettings | None = None,
        http_client: HttpClient | None = None,
        task_timeout: int = 60,
    ) -> None:
        parameters = parameters or BaiduImageGenerationParameters()
        settings = settings or BaiduCreationSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(parameters=parameters, settings=settings, http_client=http_client)

        self.token_manager = BaiduCreationTokenManager(self.settings, self.http_client)
        self.task_timeout = task_timeout

    def _get_request_parameters(self, prompt: str, parameters: BaiduImageGenerationParameters) -> HttpxPostKwargs:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        json_data = {
            'prompt': prompt,
            **parameters.custom_model_dump(),
        }
        return {
            'url': self.settings.image_generation_api,
            'json': json_data,
            'headers': headers,
            'params': {
                'access_token': self.token_manager.token,
            },
        }

    @override
    def generate(self, prompt: str, **kwargs: Unpack[BaiduImageGenerationParametersDict]) -> ImageGenerationOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = self.http_client.post(request_parameters=request_parameters)
        task_id = response.json()['data']['task_id']
        image_urls = self._get_image_urls(task_id)
        generated_images: list[GeneratedImage] = []
        for image_url in image_urls:
            image = GeneratedImage(
                url=image_url,
                prompt=prompt,
                image_format='png',
                content=self.http_client.get({'url': image_url}).content,
            )
            generated_images.append(image)
        return ImageGenerationOutput(model_info=self.model_info, cost=0.3 * (parameters.n or 1), images=generated_images)

    @override
    async def async_generate(self, prompt: str, **kwargs: Unpack[BaiduImageGenerationParametersDict]) -> ImageGenerationOutput:
        parameters = self.parameters.clone_with_changes(**kwargs)
        request_parameters = self._get_request_parameters(prompt, parameters)
        response = await self.http_client.async_post(request_parameters=request_parameters)
        image_urls = await self._async_get_image_urls(response.json()['data']['task_id'])
        images: list[GeneratedImage] = []
        images_response = await asyncio.gather(*(self.http_client.async_get({'url': image_url}) for image_url in image_urls))
        for image_url, image_response in zip(image_urls, images_response):
            image = GeneratedImage(
                url=image_url,
                prompt=prompt,
                image_format='png',
                content=image_response.content,
            )
            images.append(image)
        return ImageGenerationOutput(model_info=self.model_info, cost=0.3 * (parameters.n or 1), images=images)

    def _get_image_request_parameters(self, task_id: str) -> HttpxPostKwargs:
        return {
            'url': 'https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImgv2',
            'params': {
                'access_token': self.token_manager.token,
            },
            'headers': {'Content-Type': 'application/json'},
            'json': {'task_id': str(task_id)},
        }

    def _parse_task_info(self, task_info: dict[str, Any]) -> list[str] | None:
        task_status = task_info['data']['task_status']
        if task_status == 'FAILED':
            raise UnexpectedResponseError(task_info, 'Task failed')
        if task_status == 'SUCCESS':
            image_urls: list[str] = []
            for sub_result in task_info['data']['sub_task_result_list']:
                image_url = sub_result['final_image_list'][0]['img_url']
                image_urls.append(image_url)
            return image_urls
        return None

    def _get_image_urls(self, task_id: str) -> list[str]:
        start_time = time.time()
        task_info = None
        while (time.time() - start_time) < self.task_timeout:
            response = self.http_client.post(self._get_image_request_parameters(task_id))
            task_info = response.json()
            image_urls = self._parse_task_info(task_info)
            if image_urls:
                return image_urls
            time.sleep(1)
        raise UnexpectedResponseError(task_info or {}, 'Timeout')

    async def _async_get_image_urls(self, task_id: str) -> list[str]:
        start_time = time.time()
        task_info = None
        while (time.time() - start_time) < self.task_timeout:
            response = await self.http_client.async_post(self._get_image_request_parameters(task_id))
            task_info = response.json()
            image_urls = self._parse_task_info(task_info)
            if image_urls:
                return image_urls
            await asyncio.sleep(1)
        raise UnexpectedResponseError(task_info or {}, 'Timeout')

    @property
    @override
    def name(self) -> str:
        return 'getImgv2'

    @classmethod
    @override
    def from_name(cls, name: str) -> Self:
        if name and name != 'getImgv2':
            raise ValueError(f'Invalid model name: {name}, expected: getImgv2')
        return cls()
