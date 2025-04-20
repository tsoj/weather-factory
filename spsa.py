from cutechess import CutechessMan
from dataclasses import dataclass
from random import randint
import copy
from typing import Optional, List, Dict, Any, Union


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


@dataclass
class SpsaParams:
    a: float  # Step size scaling
    c: float  # Perturbation size scaling
    A: int    # Stability constant
    alpha: float = 0.601  # Step size decay rate
    gamma: float = 0.101  # Perturbation decay rate


class SpsaTuner:
    """
    Implementation of the Simultaneous Perturbation Stochastic Approximation algorithm
    for parameter optimization.
    """
    def __init__(
        self,
        spsa_params: SpsaParams,
        params: List[Param],
        runner: Any  # CutechessMan or compatible runner
    ):
        self.params = params
        self.spsa = spsa_params
        self.runner = runner
        self.delta = [0] * len(params)
        self.t = 0  # Counts total games played
        self.initial_params = None  # Will store reference to initial params if needed

    def step(self):
        """Perform one SPSA optimization step"""
        # Update t based on games per iteration
        self.t += self.runner.games

        # Calculate SPSA coefficients for this iteration
        a_t = self.spsa.a / (self.t + self.spsa.A) ** self.spsa.alpha
        c_t = self.spsa.c / self.t ** self.spsa.gamma

        # Generate random perturbation directions (±1)
        self.delta = [randint(0, 1) * 2 - 1 for _ in range(len(self.params))]

        # Create parameter sets for + and - perturbations
        params_plus = []
        params_minus = []
        for param, delta in zip(self.params, self.delta):
            # Scale perturbation by parameter's step size and c_t
            step = delta * param.step * c_t

            # Create copies with positive and negative perturbations
            param_plus = copy.deepcopy(param)
            param_minus = copy.deepcopy(param)

            param_plus.update(step)
            param_minus.update(-step)

            params_plus.append(param_plus)
            params_minus.append(param_minus)

        # Estimate gradient by running a match
        gradient = self.estimate_gradient(params_plus, params_minus)

        # Update parameters based on gradient estimate
        for param, delta in zip(self.params, self.delta):
            # Calculate parameter-specific gradient and update
            param_grad = gradient / (delta * c_t)
            param.update(-param_grad * a_t * param.step)

    def estimate_gradient(self, params_plus: List[Param], params_minus: List[Param]) -> float:
        """
        Estimate gradient by running a match between two parameter sets.
        Returns gradient based on match outcome.
        """
        # Convert parameters to UCI format
        uci_plus = [p.as_uci for p in params_plus]
        uci_minus = [p.as_uci for p in params_minus]

        # Run match and get result
        result = self.runner.run(uci_plus, uci_minus)

        # Calculate gradient: loss - win is the raw gradient
        # (negative = parameter set A is better)
        return (result.l - result.w)
