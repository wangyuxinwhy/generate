from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from pytest_mock import MockerFixture

from generate.utils import fetch_data


def test_fetch_data_local_file_exists(tmp_path: Path) -> None:
    file_path = tmp_path / 'test.txt'
    file_path.write_text('Test data')

    data = fetch_data(str(file_path))

    assert data == b'Test data'


def test_fetch_data_local_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        fetch_data('file:///path/to/nonexistent/file.txt')


def test_fetch_data_http(mocker: MockerFixture) -> None:
    mock_httpx_get = mocker.patch('httpx.get')
    mock_httpx_get.return_value = MagicMock(content=b'Test data')

    data = fetch_data('http://example.com/test.txt')

    assert data == b'Test data'
    mock_httpx_get.assert_called_once_with('http://example.com/test.txt')


def test_fetch_data_https(mocker: MockerFixture) -> None:
    mocker.patch('httpx.get')
    httpx.get.return_value = MagicMock(content=b'Test data')

    data = fetch_data('https://example.com/test.txt')

    assert data == b'Test data'
    httpx.get.assert_called_once_with('https://example.com/test.txt')


def test_fetch_data_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match='Unsupported URL scheme ftp'):
        fetch_data('ftp://example.com/test.txt')
