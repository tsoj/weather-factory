import numpy as np
from scipy.stats import lognorm
from typing import List, Optional, Tuple

from spsa import Param


class ParamGenerator:
    """
    Handles generation of artificial parameters and simulation configurations.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed."""
        self.rng = np.random.default_rng(seed)

        # Default lognormal distribution parameters
        self.lognorm_shape = 2.583
        self.lognorm_loc = 0.0
        self.lognorm_scale = 43.6152

    def generate_lognormal_sample(self, min_value: float = 1.0) -> float:
        """Generate a sample from the lognormal distribution with minimum value."""
        sample = lognorm.rvs(
            s=self.lognorm_shape,
            loc=self.lognorm_loc,
            scale=self.lognorm_scale,
            size=1,
            random_state=self.rng
        )[0]
        return max(min_value, sample)

    def generate_parameters(
        self,
        num_params: int,
        min_val: float,
        max_val: float,
        rel_step_size: float = 0.1
    ) -> List[Param]:
        """
        Generate a list of artificial parameters.

        Args:
            num_params: Number of parameters to generate
            min_val: Minimum allowable parameter value
            max_val: Maximum allowable parameter value
            rel_step_size: Step size as fraction of parameter value (default 1/10)

        Returns:
            List of generated Param objects
        """
        params = []

        for i in range(num_params):
            name = f"ArtParam_{i}"

            # Generate start value from lognormal distribution
            start_val = self.generate_lognormal_sample()

            # Clamp start value to allowed range
            start_val = min(max(start_val, min_val), max_val)

            # Calculate step size as fraction of start value
            step = max(1.0, start_val * rel_step_size)

            # Create parameter (value starts at start_val)
            param = Param(
                name=name,
                value=start_val,
                min_value=min_val,
                max_value=max_val,
                step=step,
                start_val=start_val
            )

            params.append(param)

        return params

    def generate_optimal_values(
        self,
        params: List[Param],
        rel_change: float = 0.1
    ) -> Tuple[List[float], List[float]]:
        """
        Generate optimal values and influence weights for parameters.

        Args:
            params: List of parameters to generate optimal values for
            rel_change: Relative size of change for optimal values (default 1/10)

        Returns:
            Tuple of (optimal_values, influence_weights)
        """
        optimal_values = []
        influences = []

        for param in params:
            # Generate a random influence weight (0-1)
            influence = self.rng.uniform(0.0, 1.0)
            influences.append(influence)

            # Generate modification amount based on lognormal distribution
            change_amount = self.generate_lognormal_sample() * rel_change

            # Randomly decide if change is positive or negative
            if self.rng.random() < 0.5:
                change_amount = -change_amount

            # Calculate optimal value as start value plus change
            optimal_value = param.start_val + change_amount

            # Ensure optimal value is within parameter bounds
            optimal_value = min(max(optimal_value, param.min_value), param.max_value)

            optimal_values.append(optimal_value)

        return optimal_values, influences
