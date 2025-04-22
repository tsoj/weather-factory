from cutechess import CutechessMan
from dataclasses import dataclass
import random
import copy
from typing import Optional, List, Dict, Any, Union
from param import Param



@dataclass
class SpsaParams:
    a: float  # Step size scaling
    c: float  # Perturbation size scaling
    A: int    # Stability constant
    alpha: float = 0.601  # Step size decay rate
    gamma: float = 0.101  # Perturbation decay rate
    adapt_frequency: int = 10      # Run step adaptation every n iterations (n)
    adapt_pert_magnitude: float = 5.0 # Factor M for perturbation (e.g., 1.5 => 1/1.5 to 1.5)
    adapt_update_decay: float = 0.99 # Multiplicative decay factor for step updates


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
        # c_t = self.spsa.c / self.t ** self.spsa.gamma

        # Generate random perturbation directions (±1)
        self.delta = [random.randint(0, 1) * 2 - 1 for _ in range(len(self.params))]

        # Create parameter sets for + and - perturbations
        params_plus = []
        params_minus = []
        for param, delta in zip(self.params, self.delta):
            # Scale perturbation by parameter's step size and c_t
            step = delta * param.step * self.spsa.c

            # Create copies with positive and negative perturbations
            param_plus = copy.deepcopy(param)
            param_minus = copy.deepcopy(param)

            param_plus.update(step)
            param_minus.update(-step)

            params_plus.append(param_plus)
            params_minus.append(param_minus)

        # Estimate gradient by running a match
        gradient = self.estimate_gradient(params_plus, params_minus)


        winner_sign = None
        if gradient > 1e-9: # B (-delta) won
            winner_sign = 1
        elif gradient < -1e-9: # A (+delta) won
            winner_sign = -1

        # --- 2. Periodic Step Size Adaptation (Conditional) ---
        if True and winner_sign is not None: # TODO use adapt frequency

            # Generate symmetric multiplicative step perturbation factors

            perturbation_factors = [random.uniform(1.0, self.spsa.adapt_pert_magnitude) for _ in self.params]
            perturbation_factors = [1.0 / f if random.randint(0, 1) == 1 else f for f in perturbation_factors]


            params_step_plus = []
            params_step_minus = []


            # Create parameter sets for the step adaptation game
            # Both start from the *current* base parameters (self.uci_params)
            # and apply the *winning* primary perturbation direction, scaled by perturbed steps.
            for i, param in enumerate(self.params):
                # Winning primary perturbation direction for this parameter
                # delta_i is the random +/- 1, winner_sign adjusts based on who won
                winning_primary_delta_direction = self.delta[i] * winner_sign

                # Calculate the shift using the perturbed step sizes
                # Shift = direction * base_step * c_t * perturbation_factor
                shift_plus = winning_primary_delta_direction * param.step * self.spsa.c * perturbation_factors[i]
                shift_minus = winning_primary_delta_direction * param.step  * self.spsa.c * 1.0/perturbation_factors[i]

                param_plus = copy.deepcopy(param)
                param_minus = copy.deepcopy(param)

                param_plus.update(shift_plus)
                param_minus.update(shift_minus)

                params_step_plus.append(param_plus)
                params_step_minus.append(param_minus)

            # Run the step adaptation game
            step_game_result = self.runner.run(params_step_plus, params_step_minus)
            step_gradient_signal = float(step_game_result.l - step_game_result.w) # Loss(Plus) - Win(Plus)

            # Determine which step size perturbation won
            # If > 0, Minus won (smaller steps better)
            # If < 0, Plus won (larger steps better)
            if step_gradient_signal > 1e-9: # Minus won (smaller steps better)
                step_update_direction = -1
            elif step_gradient_signal < -1e-9: # Plus won (larger steps better)
                step_update_direction = 1
            else: # Tie
                step_update_direction = None

            # Update step sizes (use bounded update)
            if step_update_direction is not None:
                for i, param in enumerate(self.params):
                    factor = perturbation_factors[i] if step_update_direction == 1 else 1.0 / perturbation_factors[i]
                    new_step = param.step * self.spsa.adapt_update_decay
                    new_step += (1.0 - self.spsa.adapt_update_decay) * factor * param.step
                    param.update_step(new_step) # Applies bounds


        # Update parameters based on gradient estimate
        for param, delta in zip(self.params, self.delta):
            # Calculate parameter-specific gradient and update
            param_grad = gradient / (delta * self.spsa.c)
            param.update(-param_grad * a_t * param.step)

    def estimate_gradient(self, params_plus: List[Param], params_minus: List[Param]) -> float:
        """
        Estimate gradient by running a match between two parameter sets.
        Returns gradient based on match outcome.
        """
        # Convert parameters to UCI format

        # Run match and get result
        result = self.runner.run(params_plus, params_minus)

        # Calculate gradient: loss - win is the raw gradient
        # (negative = parameter set A is better)
        return (result.l - result.w)
