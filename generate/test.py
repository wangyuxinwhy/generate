from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence, Type

import pytest

from generate.model import ModelParameters

param_type = Literal['model', 'model_cls', 'parameter', 'parameter_cls']


def get_pytest_params(
    id_prefix: str,
    model_registry: Mapping[str, tuple[Any, Type[ModelParameters]]],
    types: Sequence[param_type] | param_type,
    exclude: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
) -> list[Any]:
    exclude = exclude or []
    include = include
    if isinstance(types, str):
        types = [types]

    pytest_params: list[Any] = []
    for model_name, (model_cls, paramter_cls) in model_registry.items():
        if model_name in exclude:
            continue
        if include and model_name not in include:
            continue
        values: list[Any] = []
        for t in types:
            if t == 'model':
                values.append(model_cls(parameters=paramter_cls()))
            elif t == 'model_cls':
                values.append(model_cls)
            elif t == 'parameter':
                values.append(paramter_cls())
            elif t == 'parameter_cls':
                values.append(paramter_cls)
            else:
                raise ValueError(f'Unknown type {t}')
        pytest_params.append(pytest.param(*values, id=f'{id_prefix}_{model_name}'))
    return pytest_params
