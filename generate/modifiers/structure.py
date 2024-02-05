from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar

from pydantic import BaseModel
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
from generate.model import GenerateModel, ModelOutput, ModelParametersDict

field_info_title = 'Output JSON strictly based the format and pydantic field information below:\n'
json_schema_title = 'Output JSON strictly based the OpenAI JSON Schema:\n'
system_template = """\
# Instruction
{instruction}
# Output Format
{output_format_description}
"""


I = TypeVar('I', BaseModel, str, Dict[str, Any])  # noqa: E741
O = TypeVar('O', bound=BaseModel)  # noqa: E741
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


class StructureKwargs(TypedDict, Generic[O], total=False):
    examples: Optional[Iterable[Example[O]]]
    system_template: str
    max_num_reask: int


class Structure(GenerateModel[str, StructureModelOutput[O]], Generic[M, O]):
    def __init__(
        self,
        model: M,
        instruction: str,
        output_structure_type: Type[O],
        examples: Optional[Iterable[Example[O]]] = None,
        system_template: str = system_template,
        max_num_reask: int = 2,
    ) -> None:
        self.model = model
        self.instruction = instruction
        self.output_structure_type = output_structure_type
        self.examples = examples or []
        self.system_template = system_template
        self.max_num_reask = max_num_reask

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
            messages.append(AssistantMessage(content=example.output.model_dump_json(exclude_none=True)))
        return messages

    @property
    def _output_format_description(self) -> str:
        json_schema = self.output_structure_type.model_json_schema()
        have_ref = '$defs' in json_schema
        if have_ref:
            text = json_schema_title
            json_schema = json.dumps(json_schema, indent=2)
            text += json_schema
            return text
        text = field_info_title
        fields_info = str(self.output_structure_type.model_fields)
        fields_info = fields_info.replace("'", '"')
        text += fields_info
        return text

    @property
    def system_message(self) -> SystemMessage:
        system_content = self.system_template.format(
            instruction=self.instruction, output_format_description=self._output_format_description
        )
        return SystemMessage(content=system_content)

    def generate(self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]) -> StructureModelOutput[O]:
        messages = deepcopy(self.messages)
        messages.extend(ensure_messages(prompt))
        num_reask = 0
        cost = None
        while num_reask <= self.max_num_reask:
            model_output = self.model.generate(messages, **kwargs)
            messages.append(model_output.message)
            if model_output.cost is not None:
                if cost is None:
                    cost = model_output.cost
                else:
                    cost += model_output.cost

            try:
                json_string = ensure_valid_json(model_output.reply)
                structure = self.output_structure_type.model_validate_json(json_string)
                return StructureModelOutput(
                    model_info=model_output.model_info, structure=structure, cost=cost, extra=model_output.extra
                )
            except Exception as e:
                num_reask += 1
                messages.append(
                    UserMessage(content=f'I got an error, please try to fix your output based on the error message. Error: {e}')
                )

        raise ValueError(f'Failed to generate valid JSON after {self.max_num_reask} reasks.')

    async def async_generate(self, prompt: Prompt, **kwargs: Unpack[ModelParametersDict]) -> StructureModelOutput[O]:
        messages = deepcopy(self.messages)
        messages.extend(ensure_messages(prompt))
        num_reask = 0
        cost = None
        while num_reask <= self.max_num_reask:
            model_output = await self.model.async_generate(messages, **kwargs)
            messages.append(model_output.message)
            if model_output.cost is not None:
                if cost is None:
                    cost = model_output.cost
                else:
                    cost += model_output.cost

            try:
                json_string = ensure_valid_json(model_output.reply)
                structure = self.output_structure_type.model_validate_json(json_string)
                return StructureModelOutput(
                    model_info=model_output.model_info, structure=structure, cost=cost, extra=model_output.extra
                )
            except Exception as e:
                num_reask += 1
                messages.append(
                    UserMessage(content=f'I got an error, please try to fix your output based on the error message. Error: {e}')
                )

        raise ValueError(f'Failed to generate valid JSON after {self.max_num_reask} reasks.')
