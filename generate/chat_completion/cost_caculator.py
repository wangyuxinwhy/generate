from __future__ import annotations

from typing import Protocol

from generate.types import ModelPrice


class CostCalculator(Protocol):
    def calculate(self, model_name: str, input_tokens: int, output_tokens: int) -> float | None:
        ...


class GeneralCostCalculator(CostCalculator):
    def __init__(self, model_price: ModelPrice, exchange_rate: float = 1) -> None:
        # per million tokens
        self.model_price = model_price
        self.exchange_rate = exchange_rate

    def calculate(self, model_name: str, input_tokens: int, output_tokens: int) -> float | None:
        if self.model_price is None:
            return None
        for model, (input_token_price, output_token_price) in self.model_price.items():
            if model in model_name:
                cost = input_token_price * (input_tokens / 1_000_000) + output_token_price * (output_tokens / 1_000_000)
                return cost * self.exchange_rate
        return None
