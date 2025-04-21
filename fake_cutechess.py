import random
import math
import re
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from param import Param
from cutechess import MatchResult


@dataclass
class FakeSimConfig:
    """Configuration for the chess engine parameter simulation."""
    # List of optimal parameter values
    optimal_values: List[float]
    # How much each parameter influences engine strength (0-1)
    parameter_influences: List[float]
    # Elo advantage of the optimal parameters compared to the initial parameters
    base_elo_advantage: float = 20.0
    # Base draw rate for evenly matched positions
    base_draw_rate: float = 0.60
    # How much draw rate decreases per Elo difference
    draw_rate_elo_sensitivity: float = 0.01
    # Calculated fields
    initial_total_weighted_deviation: float = field(init=False, default=0.0)
    elo_per_deviation_unit: float = field(init=False, default=0.0)

    def calculate_scaling(self, initial_params: List[Param]):
        """Calculate initial deviation and Elo scaling factor."""
        total_dev = 0.0
        print("Calculating initial deviation for Elo scaling:")

        for i, param in enumerate(initial_params):
            optimal_value = self.optimal_values[i]
            influence = self.parameter_influences[i]

            # Calculate deviation between initial and optimal values
            deviation = abs(param.start_val - optimal_value)
            weighted_dev = influence * deviation
            total_dev += weighted_dev

            print(f"  Param '{param.name}': Initial={param.start_val:.3f}, "
                  f"Optimal={optimal_value:.3f}, "
                  f"Influence={influence:.3f}, WeightedDev={weighted_dev:.3f}")

        self.initial_total_weighted_deviation = total_dev
        print(f"  Total Initial Weighted Deviation: {self.initial_total_weighted_deviation:.4f}")

        if self.initial_total_weighted_deviation < 1e-9:
            # Initial parameters are already optimal
            self.elo_per_deviation_unit = 0.0
            print("  Initial parameters are optimal. Elo scaling factor set to 0.")
        else:
            self.elo_per_deviation_unit = self.base_elo_advantage / self.initial_total_weighted_deviation
            print(f"  Elo Scaling Factor (Elo per unit deviation): {self.elo_per_deviation_unit:.4f}")



def calculate_total_weighted_deviation(sim_config: FakeSimConfig, current_params: List[Param]) -> float:
    """
    Calculate how far the current parameters are from the optimal values.
    Lower is better (closer to optimal).
    """
    total_deviation = 0.0

    for i, current_param in enumerate(current_params):
        influence = sim_config.parameter_influences[i]
        optimal_value = sim_config.optimal_values[i]

        # Calculate deviation from the optimal value
        deviation = abs(round(current_param.value) - round(optimal_value))
        weighted_deviation = influence * deviation
        total_deviation += weighted_deviation

    return total_deviation

class FakeCutechessMan:
    """
    Simulates chess engine match results based on parameter values' proximity
    to a hidden optimal configuration.
    """
    def __init__(
        self,
        initial_params: List[Param],
        sim_config: FakeSimConfig,
        games: int = 32,
        engine_name: str = "fake_engine",
        book: str = "fake_book.epd",
        tc: float = 5.0,
        hash_size: int = 8,
        threads: int = 1,
        save_rate: int = 10,  # Not used in simulation
        pgnout: str = "tuner/fake_games.pgn",  # Not used in simulation
        use_fastchess: bool = True  # Not used in simulation
    ):
        # Store initial parameters
        self.initial_params = copy.deepcopy(initial_params)
        self.sim_config = sim_config
        self.games = games
        self.engine_name = engine_name
        self.param_names = [p.name for p in self.initial_params]

        # Validate configuration
        if len(self.initial_params) != len(sim_config.optimal_values):
            raise ValueError("Mismatch between initial_params and optimal_values length")
        if len(self.initial_params) != len(sim_config.parameter_influences):
            raise ValueError("Mismatch between initial_params and parameter_influences length")

        # Calculate scaling factor for Elo calculations
        self.sim_config.calculate_scaling(self.initial_params)

        # Print simulation configuration
        print("--- FakeCutechess Simulation Initialized ---")
        print(f"Simulating {self.games} games per match.")
        print("Target Optimal Values:")
        for (name, optimal) in zip(self.param_names, self.sim_config.optimal_values):
            print(f"  {name}: {optimal:.3f}")
        print("Parameter Influences:")
        for (name, influence) in zip(self.param_names, self.sim_config.parameter_influences):
            print(f"  {name}: {influence:.3f}")
        print(f"Target Elo Advantage (Optimal vs Initial): {self.sim_config.base_elo_advantage}")
        print(f"Initial Total Weighted Deviation: {self.sim_config.initial_total_weighted_deviation:.4f}")
        print(f"Elo Scaling Factor: {self.sim_config.elo_per_deviation_unit:.4f} Elo / unit deviation")
        print(f"Base Draw Rate: {self.sim_config.base_draw_rate}")
        print(f"Draw Rate Elo Sensitivity: {self.sim_config.draw_rate_elo_sensitivity}")
        print("------------------------------------------")

    def _get_elo_diff(self, deviation_a: float, deviation_b: float) -> float:
        """Convert difference in deviations to Elo difference."""
        # Smaller deviation means stronger engine
        elo_diff = (deviation_b - deviation_a) * self.sim_config.elo_per_deviation_unit
        return elo_diff

    def _get_probabilities(self, elo_diff: float) -> Tuple[float, float, float]:
        """Calculate win/loss/draw probabilities based on Elo difference."""
        # Expected score for side A based on Elo
        expected_score_a = 1 / (1 + 10**(-elo_diff / 400))

        # Calculate draw rate based on closeness of match
        draw_prob = self.sim_config.base_draw_rate * math.exp(
            -self.sim_config.draw_rate_elo_sensitivity * abs(elo_diff)
        )
        draw_prob = max(0, min(1, draw_prob))  # Clamp between 0 and 1

        # Calculate actual win/loss probabilities
        win_prob_a = expected_score_a - 0.5 * draw_prob
        loss_prob_a = (1 - expected_score_a) - 0.5 * draw_prob

        # Ensure probabilities are valid
        win_prob_a = max(0, win_prob_a)
        loss_prob_a = max(0, loss_prob_a)

        # Normalize to ensure probabilities sum to 1
        prob_sum = win_prob_a + loss_prob_a + draw_prob
        if prob_sum > 1e-9:
            win_prob_a /= prob_sum
            loss_prob_a /= prob_sum
            draw_prob /= prob_sum
        else:
            # Handle edge cases
            if elo_diff > 0:
                win_prob_a, loss_prob_a, draw_prob = 1.0, 0.0, 0.0
            elif elo_diff < 0:
                win_prob_a, loss_prob_a, draw_prob = 0.0, 1.0, 0.0
            else:
                win_prob_a, loss_prob_a, draw_prob = 0.0, 0.0, 1.0

        return win_prob_a, loss_prob_a, draw_prob

    def run(self, params_a: List[Param], params_b: List[Param]) -> MatchResult:
        """Simulate a match between two parameter sets."""

        # Calculate total deviation from optimal for each set
        deviation_a = calculate_total_weighted_deviation(self.sim_config, params_a)
        deviation_b = calculate_total_weighted_deviation(self.sim_config, params_b)

        # Calculate Elo difference
        elo_diff = self._get_elo_diff(deviation_a, deviation_b)

        # Get outcome probabilities
        win_a, win_b, draw = self._get_probabilities(elo_diff)

        # Simulate games
        wins_a = 0
        wins_b = 0
        draws = 0

        for _ in range(self.games):
            r = random.random()
            if r < win_a:
                wins_a += 1
            elif r < win_a + win_b:
                wins_b += 1
            else:
                draws += 1

        return MatchResult(w=wins_a, l=wins_b, d=draws, elo_diff=elo_diff)

    def get_cutechess_cmd(self, params_a: List[str], params_b: List[str]) -> str:
        """Return a dummy command string (for compatibility with real CutechessMan)."""
        return (f"FAKE_SIMULATION engine={self.engine_name} games={self.games} "
                f"params_a='{' '.join(params_a)}' params_b='{' '.join(params_b)}'")
