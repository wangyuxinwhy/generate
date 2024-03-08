from __future__ import annotations

import hashlib
import json
from io import BytesIO
from typing import AsyncIterator, ClassVar, Iterator, List, Literal, Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, TypedDict, Unpack, override

from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    ImagePart,
    ImageUrlPart,
    Message,
    MessageTypeError,
    Prompt,
    SystemMessage,
    TextPart,
    UserMessage,
    UserMultiPartMessage,
    ensure_messages,
)
from generate.chat_completion.model_output import ChatCompletionOutput, ChatCompletionStreamOutput
from generate.chat_completion.stream_manager import StreamManager
from generate.http import (
    HttpClient,
    HttpGetKwargs,
    HttpxPostKwargs,
    ResponseValue,
)
from generate.model import ModelParameters, RemoteModelParametersDict
from generate.platforms.dashscope import DashScopeSettings
from generate.types import Probability


class DashScopeMultiModalChatParameters(ModelParameters):
    seed: Optional[PositiveInt] = None
    top_p: Optional[Probability] = Field(default=None, alias='TopP')
    top_k: Optional[Annotated[int, Field(ge=0, le=100)]] = None


class DashScopeMultiModalChatParametersDict(RemoteModelParametersDict, total=False):
    seed: Optional[PositiveInt]
    top_p: Optional[Probability]
    top_k: Optional[int]


class DashScopeMultiModalMessage(TypedDict):
    role: str
    content: list[dict[Literal['image', 'text'], str]]


class DashScopeMultiModalChat(RemoteChatCompletionModel):
    model_type: ClassVar[str] = 'dashscope_multimodal'
    available_models: ClassVar[List[str]] = ['qwen-vl-max', 'qwen-vl-plus']

    parameters: DashScopeMultiModalChatParameters
    settings: DashScopeSettings

    def __init__(
        self,
        model: str = 'qwen-vl-max',
        parameters: DashScopeMultiModalChatParameters | None = None,
        settings: DashScopeSettings | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        parameters = parameters or DashScopeMultiModalChatParameters()
        settings = settings or DashScopeSettings()  # type: ignore
        http_client = http_client or HttpClient()
        super().__init__(model=model, parameters=parameters, settings=settings, http_client=http_client)

    def upload_image(self, image: bytes, image_format: str) -> str:
        get_kwargs = HttpGetKwargs(
            url=f'{self.settings.api_base}/uploads',
            params={'action': 'getPolicy', 'model': self.model},
            headers={'Authorization': f'Bearer {self.settings.api_key.get_secret_value()}'},
        )
        response_data = self.http_client.get(get_kwargs)
        upload_info = response_data.json()['data']

        form_data = {}
        form_data['OSSAccessKeyId'] = upload_info['oss_access_key_id']
        form_data['Signature'] = upload_info['signature']
        form_data['policy'] = upload_info['policy']
        hash_code = hashlib.md5(image).hexdigest()
        form_data['key'] = upload_info['upload_dir'] + '/' + f'{hash_code}.{image_format}'
        form_data['x-oss-object-acl'] = upload_info['x_oss_object_acl']
        form_data['x-oss-forbid-overwrite'] = upload_info['x_oss_forbid_overwrite']
        form_data['success_action_status'] = '200'
        form_data['x-oss-content-type'] = f'image/{image_format}'
        url = upload_info['upload_host']
        files = {'file': BytesIO(image)}
        response = self.http_client.client.post(
            url,
            data=form_data,
            files=files,
        )
        response.raise_for_status()
        return 'oss://' + form_data['key']

    @override
    def generate(self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]) -> ChatCompletionOutput:
        return super().generate(prompt, **kwargs)

    @override
    async def async_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> ChatCompletionOutput:
        return await super().async_generate(prompt, **kwargs)

    @override
    def stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> Iterator[ChatCompletionStreamOutput]:
        yield from super().stream_generate(prompt, **kwargs)

    @override
    async def async_stream_generate(
        self, prompt: Prompt, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> AsyncIterator[ChatCompletionStreamOutput]:
        async for output in super().async_stream_generate(prompt, **kwargs):
            yield output

    def _has_oss_file(self, message: DashScopeMultiModalMessage) -> bool:
        for part in message['content']:
            for k, v in part.items():
                if k == 'image' and v.startswith('oss://'):
                    return True
        return False

    def _convert_message(self, message: Message) -> DashScopeMultiModalMessage:
        if isinstance(message, UserMessage):
            return {'role': 'user', 'content': [{'text': message.content}]}
        if isinstance(message, AssistantMessage):
            return {'role': 'assistant', 'content': [{'text': message.content}]}
        if isinstance(message, SystemMessage):
            return {'role': 'system', 'content': [{'text': message.content}]}
        if isinstance(message, UserMultiPartMessage):
            content = []
            for part in message.content:
                if isinstance(part, TextPart):
                    content.append({'text': part.text})
                elif isinstance(part, ImageUrlPart):
                    content.append({'image': part.image_url.url})
                elif isinstance(part, ImagePart):
                    image_url = self.upload_image(part.image, part.image_format or 'png')
                    content.append({'image': image_url})
                else:
                    raise TypeError(f'Unsupported part type: {part}')
            return {'role': 'user', 'content': content}
        allowed_message_type = (UserMessage, AssistantMessage, SystemMessage, UserMultiPartMessage)
        raise MessageTypeError(message, allowed_message_type=allowed_message_type)

    @override
    def _get_request_parameters(
        self, prompt: Prompt, stream: bool = False, **kwargs: Unpack[DashScopeMultiModalChatParametersDict]
    ) -> HttpxPostKwargs:
        messages = ensure_messages(prompt)
        parameters = self.parameters.clone_with_changes(**kwargs)
        dashscope_messages = [self._convert_message(message) for message in messages]
        headers = {
            'Authorization': self.settings.api_key.get_secret_value(),
            'Content-Type': 'application/json',
        }
        if any(self._has_oss_file(message) for message in dashscope_messages):
            headers['X-DashScope-OssResourceResolve'] = 'enable'
        params = {
            'input': {
                'messages': dashscope_messages,
            },
            'model': self.model,
            'parameters': parameters.custom_model_dump(),
        }
        if stream:
            headers['Accept'] = 'text/event-stream'

        return {
            'url': f'{self.settings.api_base}/services/aigc/multimodal-generation/generation',
            'headers': headers,
            'json': params,
        }

    @override
    def _process_reponse(self, response: ResponseValue) -> ChatCompletionOutput:
        choice = response['output']['choices'][0]
        content_list = choice['message']['content']
        text = ''
        result_images = []
        for content in content_list:
            for k, v in content.items():
                if k != 'result_image':
                    text += v
                else:
                    result_images.append(v)
        return ChatCompletionOutput(
            model_info=self.model_info,
            finish_reason=choice.get('finish_reason'),
            message=AssistantMessage(content=text),
            cost=None,
            extra={
                'usage': response['usage'],
                'request_id': response['request_id'],
                'content': content_list,
                'result_images': result_images,
            },
        )

    @override
    def _process_stream_line(self, line: str, stream_manager: StreamManager) -> Optional[ChatCompletionStreamOutput]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        choice = data['output']['choices'][0]
        finish_reason = choice['finish_reason']
        reply = choice['message']['content'][0]['text']
        usage = data['usage']
        request_id = data['request_id']
        extra = {
            'usage': usage,
            'response_id': request_id,
        }
        if finish_reason == 'stop':
            stream_manager.finish_reason = 'stop'
            stream_manager.delta = ''
            stream_manager.extra.update(extra)
            return stream_manager.build_stream_output()
        stream_manager.delta = reply[len(stream_manager.content) :]
        return stream_manager.build_stream_output()
