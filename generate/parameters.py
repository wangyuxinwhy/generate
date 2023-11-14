from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelParameters(BaseModel):
    def custom_model_dump(self) -> dict[str, Any]:
        return {**self.model_dump(exclude_none=True, by_alias=True), **self.model_dump(exclude_unset=True, by_alias=True)}
