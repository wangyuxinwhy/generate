from __future__ import annotations

# yuan per thousand tokens
DefaultPriceMap = {
    'moonshot': {
        'moonshot-v1-8k': (0.012, 0.012),
        'moonshot-v1-32k': (0.024, 0.024),
        'moonshot-v1-128k': (0.06, 0.06),
    },
    'minimax': {
        'abab5.5-chat': (0.015, 0.015),
        'abab5.5s-chat': (0.005, 0.005),
        'abab6-chat': (0.1, 0.1),
    },
}


class GeneralCostCalculator:
    def __init__(self, price_map: dict[str, dict[str, tuple[float, float]]] | None = None) -> None:
        self.price_map = price_map or DefaultPriceMap

    def calculate(self, model_type: str, model_name: str, input_tokens: int, output_tokens: int) -> float | None:
        if model_type in self.price_map and model_name in self.price_map[model_type]:
            price = self.price_map[model_type][model_name]
            return (input_tokens * price[0] + output_tokens * price[1]) / 1000
        return None
