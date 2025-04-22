from dataclasses import dataclass
from typing import Optional

@dataclass
class Param:
    name: str
    value: float
    min_value: float
    max_value: float
    step: float
    start_val: Optional[float] = None

    def __post_init__(self):
        if self.start_val is None:
            self.start_val = self.value
        if self.step <= 0:
            raise ValueError(f"Step must be positive, got {self.step} for param {self.name}")
        # Clamp initial and current values
        self.start_val = min(max(self.start_val, self.min_value), self.max_value)
        self.value = min(max(self.value, self.min_value), self.max_value)

    def get(self) -> int:
        return round(self.value)

    def update(self, amt: float):
        """Update parameter by amount, staying within bounds"""
        self.value = min(max(self.value + amt, self.min_value), self.max_value)

    def update_step(self, new_step: float):
        """Updates the step size, ensuring it stays within bounds."""
        self.step = max(1.0, new_step)

    @property
    def as_uci(self) -> str:
        """Format as UCI parameter string"""
        return f"option.{self.name}={self.get()}"

    def get_change(self) -> str:
        """Return formatted string showing change from start value"""
        diff = self.value - self.start_val
        if diff > 0:
            return f"+{diff:.2f}"
        elif diff < 0:
            return f"{diff:.2f}"
        else:
            return f"±0"

    def __str__(self) -> str:
        return (
            f"{self.name} = {self.get()}({self.get_change()}) in "
            f"[{self.min_value}, {self.max_value}], step={self.step:.2f}"
        )
