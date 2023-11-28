from typing import Any, Literal

from generate.chat_completion.message.core import (
    FunctionCall,
    Message,
    Messages,
    Prompt,
    UnionMessage,
    UserMessage,
    assistant_content_validator,
    message_validator,
    user_content_validator,
)


def ensure_messages(prompt: Prompt) -> Messages:
    """
    Ensure that the given prompt is in the form of a list of messages.

    Args:
        prompt (Prompt): The prompt to be validated.

    Returns:
        Messages: A list of message.

    Raises:
        ValidationError: If the prompt is not valid.
    """
    if isinstance(prompt, str):
        return [UserMessage(content=prompt)]
    if isinstance(prompt, dict):
        return [message_validator.validate_python(prompt)]
    if isinstance(prompt, Message):
        return [prompt]

    messages: list[UnionMessage] = []
    for i in prompt:
        if isinstance(i, Message):
            messages.append(i)
        else:
            messages.append(message_validator.validate_python(i))
    return messages


def infer_assistant_message_content_type(message_content: Any) -> Literal['text', 'function_call', 'tool_calls']:
    obj = assistant_content_validator.validate_python(message_content)
    if isinstance(obj, str):
        return 'text'
    if isinstance(obj, FunctionCall):
        return 'function_call'
    return 'tool_calls'


def infer_user_message_content_type(message_content: Any) -> Literal['text', 'multi_part']:
    obj = user_content_validator.validate_python(message_content)
    if isinstance(obj, str):
        return 'text'
    return 'multi_part'
