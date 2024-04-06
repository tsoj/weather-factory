from dataclasses import dataclass

@dataclass
class Param:
    name: str
    value: float
    min_value: int
    max_value: int
    step: float

    def __post_init__(self):
        self.start_val: float = self.value
        assert self.step > 0

    def get(self) -> int:
        return round(self.value)

    def update(self, amt: float):
        self.value = min(max(self.value + amt, self.min_value), self.max_value)

    def get_change(self) -> str:
        if self.value > self.start_val:
            return f"+{self.value - self.start_val:.2f}"
        elif self.value < self.start_val:
            return f"-{self.start_val - self.value:2f}"
        else:
            return f"+-0"

    def __str__(self) -> str:
        return (
            f"{self.name} = {self.get()}({self.get_change()}) in "
            f"[{self.min_value}, {self.max_value}] "
        )