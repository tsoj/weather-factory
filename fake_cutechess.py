# --- START OF FILE fake_cutechess.py ---

import random
import math
import re
from dataclasses import dataclass, field # Added field
from spsa import Param # Assuming Param definition is accessible
from cutechess import MatchResult # Reuse the result dataclass

@dataclass
class FakeSimConfig:
    """Configuration for the fake simulation."""
    optimal_offsets: list[float]
    parameter_influences: list[float]
    # Elo advantage of the optimal parameters *compared to the initial parameters*
    base_elo_advantage: float = 20.0
    base_draw_rate: float = 0.60
    draw_rate_elo_sensitivity: float = 0.01
    # Calculated fields, not set by user directly
    initial_total_weighted_deviation: float = field(init=False, default=0.0)
    elo_per_deviation_unit: float = field(init=False, default=0.0)

    def calculate_scaling(self, initial_params: list[Param]):
        """Calculates deviation of initial params and the Elo scaling factor."""
        total_dev = 0.0
        print("Calculating initial deviation for Elo scaling:")
        for i, param in enumerate(initial_params):
            # Initial offset is always 0 by definition
            optimal_offset = self.optimal_offsets[i]
            influence = self.parameter_influences[i]
            deviation = abs(0.0 - optimal_offset)
            weighted_dev = influence * deviation
            total_dev += weighted_dev
            # print(f"  Param '{param.name}': OptimalOffset={optimal_offset:+.3f}, Influence={influence:.3f}, WeightedDev={weighted_dev:.3f}")

        self.initial_total_weighted_deviation = total_dev
        print(f"  Total Initial Weighted Deviation: {self.initial_total_weighted_deviation:.4f}")

        if self.initial_total_weighted_deviation < 1e-9:
            # Initial parameters are (practically) optimal. No Elo advantage.
            self.elo_per_deviation_unit = 0.0
            print("  Initial parameters are optimal. Elo scaling factor set to 0.")
        else:
            self.elo_per_deviation_unit = self.base_elo_advantage / self.initial_total_weighted_deviation
            print(f"  Elo Scaling Factor (Elo per unit deviation): {self.elo_per_deviation_unit:.4f}")


class FakeCutechessMan:
    """
    Simulates Cutechess/Fastchess runs based on parameter proximity to a hidden optimum,
    scaling Elo based on a target difference between initial and optimal parameters.
    """
    def __init__(
        self,
        initial_params: list[Param],
        sim_config: FakeSimConfig,
        games: int = 32,
        engine_name: str = "fake_engine",
        book: str = "fake_book.epd",
        tc: float = 5.0,
        hash_size: int = 8,
        threads: int = 1,
        save_rate: int = 10, # Not used by fake runner
        pgnout: str = "tuner/fake_games.pgn", # Not used by fake runner
        use_fastchess: bool = True # Not used by fake runner
    ):
        # Store a deep copy of initial params to ensure start_val is preserved
        self.initial_params = copy.deepcopy(initial_params)
        self.sim_config = sim_config
        self.games = games
        self.engine_name = engine_name
        self.param_names = [p.name for p in self.initial_params]

        if len(self.initial_params) != len(sim_config.optimal_offsets):
            raise ValueError("Mismatch between initial_params and optimal_offsets length")
        if len(self.initial_params) != len(sim_config.parameter_influences):
            raise ValueError("Mismatch between initial_params and parameter_influences length")

        # Calculate the scaling factor based on the provided initial parameters
        self.sim_config.calculate_scaling(self.initial_params)

        print("--- FakeCutechess Simulation Initialized ---")
        print(f"Simulating {self.games} games per match.")
        print("Target Optimal Offsets:")
        for name, offset in zip(self.param_names, self.sim_config.optimal_offsets):
            print(f"  {name}: {offset:+.3f}")
        print("Parameter Influences:")
        for name, influence in zip(self.param_names, self.sim_config.parameter_influences):
             print(f"  {name}: {influence:.3f}")
        print(f"Target Elo Advantage (Optimal vs Initial): {self.sim_config.base_elo_advantage}")
        print(f"Initial Total Weighted Deviation: {self.sim_config.initial_total_weighted_deviation:.4f}")
        print(f"Elo Scaling Factor: {self.sim_config.elo_per_deviation_unit:.4f} Elo / unit deviation")
        print(f"Base Draw Rate: {self.sim_config.base_draw_rate}")
        print(f"Draw Rate Elo Sensitivity: {self.sim_config.draw_rate_elo_sensitivity}")
        print("------------------------------------------")


    def _parse_uci_params(self, uci_strings: list[str]) -> dict[str, float]:
        """Parses 'option.Name=Value' strings into a dictionary."""
        parsed_params = {}
        pattern = re.compile(r"option\.([\w\s]+?)\s*=\s*(-?[\d\.]+)")
        for uci_str in uci_strings:
            match = pattern.match(uci_str)
            if match:
                name = match.group(1).strip()
                try:
                    value = float(match.group(2))
                    # Find the corresponding initial Param object to get start_val
                    initial_param = next((p for p in self.initial_params if p.name == name), None)
                    if initial_param:
                         # Store value and the initial value for offset calculation
                         parsed_params[name] = {'current': value, 'initial': initial_param.start_val}
                    else:
                         print(f"Warning: Could not find initial param data for '{name}'")
                         # Fallback: assume current value is offset from 0? Less accurate.
                         # parsed_params[name] = {'current': value, 'initial': 0.0}

                except ValueError:
                    print(f"Warning: Could not parse value in '{uci_str}'")
            else:
                 print(f"Warning: Could not parse UCI string '{uci_str}'")
        return parsed_params

    def _calculate_total_weighted_deviation(self, current_params_dict: dict[str, dict]) -> float:
        """
        Calculates the total weighted deviation of a parameter set from the optimum.
        Lower deviation is better (closer to optimum).
        """
        total_deviation = 0.0

        for i, param_config in enumerate(self.initial_params): # Iterate using initial params as reference
            name = param_config.name
            influence = self.sim_config.parameter_influences[i]
            optimal_offset = self.sim_config.optimal_offsets[i]

            param_data = current_params_dict.get(name)
            if param_data:
                current_value = param_data['current']
                initial_value = param_data['initial']
                current_offset = current_value - initial_value
            else:
                # If parameter wasn't found in UCI strings (shouldn't happen if SPSA sends all)
                # assume it hasn't changed from initial, so offset is 0
                current_offset = 0.0
                # print(f"Warning: Param '{name}' not found in parsed UCI dict, assuming offset 0.")


            # Deviation from the optimal offset
            deviation = abs(current_offset - optimal_offset)
            total_deviation += influence * deviation

        return total_deviation


    def _get_elo_diff(self, deviation_a: float, deviation_b: float) -> float:
        """Converts the difference in total weighted deviations to an Elo difference."""
        # Elo_Diff(A, B) = (Deviation(B) - Deviation(A)) * Elo_Scale
        elo_diff = (deviation_b - deviation_a) * self.sim_config.elo_per_deviation_unit
        return elo_diff

    # --- _get_probabilities remains the same ---
    def _get_probabilities(self, elo_diff: float) -> tuple[float, float, float]:
        """Calculates win/loss/draw probabilities based on Elo difference."""
        # Expected score for A based on Elo
        expected_score_a = 1 / (1 + 10**(-elo_diff / 400))

        # Calculate dynamic draw rate based on Elo difference
        draw_prob = self.sim_config.base_draw_rate * math.exp(
            -self.sim_config.draw_rate_elo_sensitivity * abs(elo_diff)
        )
        draw_prob = max(0, min(1, draw_prob)) # Clamp between 0 and 1

        # Derive win probabilities from expected score and draw probability
        win_prob_a = expected_score_a - 0.5 * draw_prob
        loss_prob_a = (1 - expected_score_a) - 0.5 * draw_prob

        # Ensure probabilities are non-negative and sum ~1
        win_prob_a = max(0, win_prob_a)
        loss_prob_a = max(0, loss_prob_a)
        prob_sum = win_prob_a + loss_prob_a + draw_prob
        if prob_sum > 1e-9:
            win_prob_a /= prob_sum
            loss_prob_a /= prob_sum
            draw_prob /= prob_sum
        else:
            if elo_diff > 0: win_prob_a = 1.0; loss_prob_a = 0.0; draw_prob = 0.0
            elif elo_diff < 0: win_prob_a = 0.0; loss_prob_a = 1.0; draw_prob = 0.0
            else: win_prob_a = 0.0; loss_prob_a = 0.0; draw_prob = 1.0

        return win_prob_a, loss_prob_a, draw_prob

    def run(self, params_a_uci: list[str], params_b_uci: list[str]) -> MatchResult:
        """Simulates a match between two parameter sets."""
        # 1. Parse UCI parameters to get current values and initial values
        params_a_dict = self._parse_uci_params(params_a_uci)
        params_b_dict = self._parse_uci_params(params_b_uci)

        # 2. Calculate total weighted deviation for each parameter set
        deviation_a = self._calculate_total_weighted_deviation(params_a_dict)
        deviation_b = self._calculate_total_weighted_deviation(params_b_dict)

        # 3. Calculate Elo difference (A vs B) based on deviations
        elo_diff = self._get_elo_diff(deviation_a, deviation_b)

        # 4. Get win/loss/draw probabilities
        win_a, win_b, draw = self._get_probabilities(elo_diff)

        # 5. Simulate `self.games` games
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

        result = MatchResult(w=wins_a, l=wins_b, d=draws, elo_diff=elo_diff)

        # Optional detailed print for debugging simulation steps
        # print(f"SimRun: DevA={deviation_a:.4f}, DevB={deviation_b:.4f}, EloDiff={elo_diff:.2f} -> W={result.w}, L={result.l}, D={result.d} (Probs W={win_a:.3f} L={win_b:.3f} D={draw:.3f})")

        return result

    # --- get_cutechess_cmd remains the same ---
    def get_cutechess_cmd(self, params_a: list[str], params_b: list[str]) -> str:
        """Returns a dummy command string, as no external process is run."""
        return f"FAKE_SIMULATION engine={self.engine_name} games={self.games} params_a='{' '.join(params_a)}' params_b='{' '.join(params_b)}'"

# --- Need copy for deepcopy ---
import copy

# --- END OF FILE fake_cutechess.py ---
