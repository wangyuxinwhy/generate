from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, ClassVar, Dict, Generic, Iterable, List, Optional, Type, TypeVar

from pydantic import BaseModel, TypeAdapter
from typing_extensions import Self, TypedDict, Unpack

from generate.chat_completion import ChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    Prompt,
    SystemMessage,
    UnionMessage,
    UserMessage,
    ensure_messages,
)
from generate.model import GenerateModel, ModelOutput, RemoteModelParametersDict

system_template = """\
# Instruction
{instruction}
# JSON Schema
{json_schema}
"""


I = TypeVar('I', BaseModel, str, Dict[str, Any])  # noqa: E741
O = TypeVar('O')  # noqa: E741
M = TypeVar('M', bound=ChatCompletionModel)


class Example(BaseModel, Generic[O]):
    prompt: Prompt
    output: O


def is_valid_json(json_str: str) -> bool:
    try:
        json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        return False
    return True


def is_valid_python_code(code_str: str) -> bool:
    try:
        eval(code_str)
    except SyntaxError:
        return False
    return True


def extract_json(json_str: str) -> str | None:
    # markdown code pattern
    json_code_pattern = r'```json\n(.*?)```'
    match = re.search(json_code_pattern, json_str, re.DOTALL)
    if match and is_valid_json(match.group(1)):
        return match.group(1)

    code_pattern = r'```(.*?)```'
    match = re.search(code_pattern, json_str, re.DOTALL)
    if match and is_valid_json(match.group(1)):
        return match.group(1)

    inline_code_pattern = r'`(.*?)`'
    match = re.search(inline_code_pattern, json_str, re.DOTALL)
    if match and is_valid_json(match.group(1)):
        return match.group(1)

    return None


def extract_python_code(code_str: str) -> str | None:
    # markdown code pattern
    json_code_pattern = r'```python\n(.*?)```'
    match = re.search(json_code_pattern, code_str, re.DOTALL)
    if match and is_valid_python_code(match.group(1)):
        return match.group(1)

    code_pattern = r'```(.*?)```'
    match = re.search(code_pattern, code_str, re.DOTALL)
    if match and is_valid_python_code(match.group(1)):
        return match.group(1)

    inline_code_pattern = r'`(.*?)`'
    match = re.search(inline_code_pattern, code_str, re.DOTALL)
    if match and is_valid_python_code(match.group(1)):
        return match.group(1)

    return None


def ensure_valid_json(json_str: str) -> str:
    if is_valid_json(json_str):
        return json_str
    if is_valid_python_code(json_str):
        return json.dumps(eval(json_str))
    if (valid_json := extract_json(json_str)) is not None:
        return valid_json
    if (valid_code := extract_python_code(json_str)) is not None:
        return json.dumps(eval(valid_code))
    raise ValueError(f'Invalid JSON, output should be a valid JSON string. Got: {json_str}')


class StructureModelOutput(ModelOutput, Generic[O]):
    structure: O


class StructureModelKwargs(TypedDict, Generic[O], total=False):
    instruction: Optional[str]
    examples: Optional[Iterable[Example[O]]]
    system_template: str
    max_num_reask: int
    output_exclude_none: bool


class StructureGenerateModel(GenerateModel[str, StructureModelOutput[O]], Generic[M, O]):
    model_task: ClassVar[str] = 'text_to_structure'

    def __init__(
        self,
        model: M,
        output_structure_type: Type[O] | TypeAdapter[O],
        instruction: str | None = None,
        examples: Optional[Iterable[Example[O]]] = None,
        system_template: str = system_template,
        output_exclude_none: bool = False,
        max_num_reask: int = 2,
    ) -> None:
        self.model = model

        if isinstance(output_structure_type, TypeAdapter):
            default_instruction = 'Extract Information'
        else:
            default_instruction = f'Extract {output_structure_type.__name__}'
        default_instruction = 'According to the JSON Schema below, parse the input text.'
        self.instruction = instruction or default_instruction

        self.output_structure_type = output_structure_type
        self.examples = examples or []
        self.system_template = system_template
        self.max_num_reask = max_num_reask
        self.output_exclude_none = output_exclude_none

        self.model_type = self.model.model_type  # type: ignore

    @property
    def is_typeadapter(self) -> bool:
        return isinstance(self.output_structure_type, TypeAdapter)

    @property
    def name(self) -> str:
        return self.model.name

    @classmethod
    def from_name(cls, name: str) -> Self:
        raise ValueError('Structure model cannot be created from name')

    @property
    def messages(self) -> List[UnionMessage]:
        messages = []
        messages.append(self.system_message)
        for example in self.examples:
            messages.extend(ensure_messages(example.prompt))
            messages.append(AssistantMessage(content=self.model_dump_json(example.output)))
        return messages

    @property
    def system_message(self) -> SystemMessage:
        system_content = self.system_template.format(
            instruction=self.instruction,
            json_schema=json.dumps(self.model_json_schema(), indent=2, ensure_ascii=False),
        )
        return SystemMessage(content=system_content)

    def model_dump_json(self, item: O) -> str:
        if self.is_typeadapter:
            assert isinstance(self.output_structure_type, TypeAdapter)
            return self.output_structure_type.dump_json(item, exclude_none=self.output_exclude_none).decode('utf-8')
        assert isinstance(item, BaseModel)
        return item.model_dump_json(exclude_none=self.output_exclude_none)

    def model_json_schema(self) -> dict[str, Any]:
        if self.is_typeadapter:
            assert isinstance(self.output_structure_type, TypeAdapter)
            return self.output_structure_type.json_schema()
        return self.output_structure_type.model_json_schema()  # type: ignore

    def model_validate_json(self, json_string: str) -> O:
        if self.is_typeadapter:
            assert isinstance(self.output_structure_type, TypeAdapter)
            return self.output_structure_type.validate_json(json_string)
        return self.output_structure_type.model_validate_json(json_string)  # type: ignore

    def generate(self, prompt: Prompt, **kwargs: Unpack[RemoteModelParametersDict]) -> StructureModelOutput[O]:
        messages = deepcopy(self.messages)
        messages.extend(ensure_messages(prompt))
        num_reask = 0
        while num_reask <= self.max_num_reask:
            model_output = self.model.generate(messages, **kwargs)
            messages.append(model_output.message)
            try:
                json_string = ensure_valid_json(model_output.reply)
                structure = self.model_validate_json(json_string)
                return StructureModelOutput(model_info=model_output.model_info, structure=structure, extra=model_output.extra)
            except Exception as e:
                num_reask += 1
                messages.append(
                    UserMessage(content=f'I got an error, please try to fix your output based on the error message. Error: {e}')
                )

        raise ValueError(f'Failed to generate valid JSON after {self.max_num_reask} reasks.')

    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[RemoteModelParametersDict]) -> StructureModelOutput[O]:
        messages = deepcopy(self.messages)
        messages.extend(ensure_messages(prompt))
        num_reask = 0
        while num_reask <= self.max_num_reask:
            model_output = await self.model.async_generate(messages, **kwargs)
            messages.append(model_output.message)
            try:
                json_string = ensure_valid_json(model_output.reply)
                structure = self.model_validate_json(json_string)
                return StructureModelOutput(model_info=model_output.model_info, structure=structure, extra=model_output.extra)
            except Exception as e:
                num_reask += 1
                messages.append(
                    UserMessage(content=f'I got an error, please try to fix your output based on the error message. Error: {e}')
                )

        raise ValueError(f'Failed to generate valid JSON after {self.max_num_reask} reasks.')
