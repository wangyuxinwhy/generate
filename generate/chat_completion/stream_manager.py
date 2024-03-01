from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from generate.chat_completion.message.core import AssistantMessage, FunctionCall, ToolCall
from generate.chat_completion.model_output import ChatCompletionStreamOutput, Stream
from generate.model import ModelInfo


class StreamManager(BaseModel):
    info: ModelInfo
    delta: Optional[str] = None
    cost: Optional[float] = None
    history_streams: List[Stream] = []
    finish_reason: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    close: bool = False
    extra: Dict[str, Any] = {}

    @property
    def already_start(self) -> bool:
        return bool(self.history_streams)

    @property
    def is_finish(self) -> bool:
        return self.finish_reason is not None

    @property
    def control(self) -> Literal['start', 'continue', 'finish']:
        if self.is_finish:
            return 'finish'
        if self.already_start:
            return 'continue'
        return 'start'

    @property
    def current_stream(self) -> Optional[Stream]:
        if self.delta is None:
            return None
        return Stream(delta=self.delta, control=self.control)

    @property
    def content(self) -> str:
        return ''.join(stream.delta for stream in self.history_streams)

    def build_stream_output(self) -> Optional[ChatCompletionStreamOutput]:
        if self.close:
            return None

        stream = self.current_stream
        if stream:
            if not self.history_streams:
                assert self.control == 'start'

            self.history_streams.append(stream)
            self.delta = None
            output = ChatCompletionStreamOutput(
                model_info=self.info,
                cost=self.cost,
                extra=self.extra,
                finish_reason=self.finish_reason,
                message=AssistantMessage(
                    content=self.content,
                    function_call=self.function_call,
                    tool_calls=self.tool_calls,
                ),
                stream=stream,
            )

            if output.is_finish:
                self.close = True
            return output

        return None
