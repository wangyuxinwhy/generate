from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, TypeVar, Union

from pydantic import Field
from typing_extensions import Annotated

T = TypeVar('T')
JsonSchema = Dict[str, Any]
Temperature = Annotated[float, Field(ge=0, le=1)]
Probability = Annotated[float, Field(ge=0, le=1)]
PrimitiveData = Optional[Union[str, int, float, bool]]
OrSequence = Union[T, Sequence[T]]
OrIterable = Union[T, Iterable[T]]
