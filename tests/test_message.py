from generate.chat_completion.message import (
    AssistantMessage,
    FunctionCall,
    FunctionCallMessage,
    UserMessage,
    ensure_messages,
)


def test_ensure_messages_with_string() -> None:
    prompt = 'Hello, how can I help you?'
    expected_messages = [UserMessage(content=prompt)]
    assert ensure_messages(prompt) == expected_messages


def test_ensure_messages_with_dict() -> None:
    prompt = {'role': 'user', 'content': 'Hello, how can I help you?'}
    expected_messages = [UserMessage(content='Hello, how can I help you?')]
    assert ensure_messages(prompt) == expected_messages

    prompt = {'role': 'assistant', 'name': 'bot', 'content': {'name': 'test', 'arguments': 'test'}}
    expected_messages = [FunctionCallMessage(role='assistant', name='bot', content=FunctionCall(name='test', arguments='test'))]
    assert ensure_messages(prompt) == expected_messages


def test_ensure_messages_with_message_types() -> None:
    prompt = UserMessage(content='Hello, how can I help you?')
    expected_messages = [UserMessage(content='Hello, how can I help you?')]
    assert ensure_messages(prompt) == expected_messages


def test_ensure_messages_with_list_of_messages() -> None:
    prompt = [
        UserMessage(content='Hello, how can I help you?'),
        AssistantMessage(content='I need help with something.'),
    ]
    expected_messages = prompt
    assert ensure_messages(prompt) == expected_messages


def test_ensure_messages_with_list_of_dicts() -> None:
    prompt = [
        {'role': 'user', 'content': 'Hello, how can I help you?'},
        {'role': 'assistant', 'content': 'I need help with something.'},
    ]
    expected_messages = [
        UserMessage(content='Hello, how can I help you?'),
        AssistantMessage(content='I need help with something.'),
    ]
    assert ensure_messages(prompt) == expected_messages
