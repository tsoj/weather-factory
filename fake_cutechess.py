# --- START OF FILE fake_cutechess.py ---

import random
import math
import re
from dataclasses import dataclass
from spsa import Param # Assuming Param definition is accessible
from cutechess import MatchResult # Reuse the result dataclass

@dataclass
class FakeSimConfig:
    """Configuration for the fake simulation."""
    # List of optimal offsets (difference from start value) for each parameter
    optimal_offsets: list[float]
    # List of influence factors (how much each param affects Elo)
    parameter_influences: list[float]
    # Base Elo difference when one player has optimal params and the other has start params
    base_elo_advantage: float = 20.0
    # Base draw rate (e.g., 0.6 means 60% draw chance if Elo diff is 0)
    base_draw_rate: float = 0.60
    # How much Elo difference reduces the draw rate (higher value = faster drop)
    draw_rate_elo_sensitivity: float = 0.01 # e.g., 100 Elo diff reduces draw rate by ~0.26

class FakeCutechessMan:
    """
    Simulates Cutechess/Fastchess runs based on parameter proximity to a hidden optimum.
    """
    def __init__(
        self,
        initial_params: list[Param],
        sim_config: FakeSimConfig,
        games: int = 32, # Number of games to simulate per 'match'
        engine_name: str = "fake_engine", # Name doesn't really matter here
        book: str = "fake_book.epd",
        tc: float = 5.0,
        hash_size: int = 8,
        threads: int = 1,
        save_rate: int = 10, # Not used by fake runner
        pgnout: str = "tuner/fake_games.pgn", # Not used by fake runner
        use_fastchess: bool = True # Not used by fake runner
    ):
        self.initial_params = initial_params # Keep track of starting values
        self.sim_config = sim_config
        self.games = games
        self.engine_name = engine_name
        self.param_names = [p.name for p in initial_params] # For parsing

        if len(self.initial_params) != len(sim_config.optimal_offsets):
            raise ValueError("Mismatch between initial_params and optimal_offsets length")
        if len(self.initial_params) != len(sim_config.parameter_influences):
            raise ValueError("Mismatch between initial_params and parameter_influences length")

        print("--- FakeCutechess Simulation Initialized ---")
        print(f"Simulating {self.games} games per match.")
        print("Target Optimal Offsets:")
        for name, offset in zip(self.param_names, self.sim_config.optimal_offsets):
            print(f"  {name}: {offset:+.3f}")
        print("Parameter Influences:")
        for name, influence in zip(self.param_names, self.sim_config.parameter_influences):
             print(f"  {name}: {influence:.3f}")
        print(f"Base Elo Advantage: {self.sim_config.base_elo_advantage}")
        print(f"Base Draw Rate: {self.sim_config.base_draw_rate}")
        print(f"Draw Rate Elo Sensitivity: {self.sim_config.draw_rate_elo_sensitivity}")
        print("------------------------------------------")


    def _parse_uci_params(self, uci_strings: list[str]) -> dict[str, float]:
        """Parses 'option.Name=Value' strings into a dictionary."""
        parsed_params = {}
        # Regex to find Name=Value, handling potential spaces around '='
        pattern = re.compile(r"option\.([\w\s]+?)\s*=\s*(-?[\d\.]+)")
        for uci_str in uci_strings:
            match = pattern.match(uci_str)
            if match:
                name = match.group(1).strip()
                try:
                    value = float(match.group(2))
                    parsed_params[name] = value
                except ValueError:
                    print(f"Warning: Could not parse value in '{uci_str}'")
            else:
                 print(f"Warning: Could not parse UCI string '{uci_str}'")
        return parsed_params

    def _calculate_score(self, current_params_dict: dict[str, float]) -> float:
        """
        Calculates a 'goodness' score based on weighted distance to optimal offsets.
        Higher score is better (closer to optimum). Score is normalized relative
        to the maximum possible deviation penalty.
        """
        total_penalty = 0.0
        max_possible_penalty = 0.0 # To normalize the score

        for i, param in enumerate(self.initial_params):
            name = param.name
            influence = self.sim_config.parameter_influences[i]
            optimal_offset = self.sim_config.optimal_offsets[i]
            initial_value = param.start_val # Use the stored start_val

            current_value = current_params_dict.get(name, initial_value) # Default to initial if not found
            current_offset = current_value - initial_value

            # Deviation from the optimal offset
            deviation = abs(current_offset - optimal_offset)
            total_penalty += influence * deviation

            # Max penalty for this param occurs at boundary farthest from optimal offset
            # TODO the provided min max values don't have any sematic meaning unter Umständen
            max_dev1 = abs((param.min_value - initial_value) - optimal_offset)
            max_dev2 = abs((param.max_value - initial_value) - optimal_offset)
            max_possible_penalty += influence * max(max_dev1, max_dev2)

        # Normalize penalty (0 = optimal, 1 = worst possible average deviation)
        normalized_penalty = total_penalty / max_possible_penalty if max_possible_penalty > 0 else 0

        # Score: Higher is better. Let's scale it so score = 0 is worst, score = 1 is optimal.
        # We relate score difference to Elo later.
        score = 1.0 - normalized_penalty
        return score


    def _get_elo_diff(self, score_a: float, score_b: float) -> float:
        """Converts the difference in scores to an Elo difference."""
        # Score difference ranges from -1 (B optimal, A worst) to +1 (A optimal, B worst)
        score_diff = score_a - score_b
        # Scale this difference by the base Elo advantage
        elo_diff = score_diff * self.sim_config.base_elo_advantage
        return elo_diff

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
        # ExpectedScore = WinProb + 0.5 * DrawProb
        # 1 - ExpectedScore = LossProb + 0.5 * DrawProb
        # WinProb + LossProb + DrawProb = 1
        win_prob_a = expected_score_a - 0.5 * draw_prob
        loss_prob_a = (1 - expected_score_a) - 0.5 * draw_prob

        # Ensure probabilities are non-negative and sum ~1 (adjust if needed due to float errors)
        win_prob_a = max(0, win_prob_a)
        loss_prob_a = max(0, loss_prob_a)
        prob_sum = win_prob_a + loss_prob_a + draw_prob
        if prob_sum > 1e-9: # Avoid division by zero
            win_prob_a /= prob_sum
            loss_prob_a /= prob_sum
            draw_prob /= prob_sum
        else: # If somehow all are zero (e.g., Elo is massive +/- infinity)
            if elo_diff > 0: win_prob_a = 1.0; loss_prob_a = 0.0; draw_prob = 0.0
            elif elo_diff < 0: win_prob_a = 0.0; loss_prob_a = 1.0; draw_prob = 0.0
            else: win_prob_a = 0.0; loss_prob_a = 0.0; draw_prob = 1.0 # Should default to draw if elo=0

        return win_prob_a, loss_prob_a, draw_prob


    def run(self, params_a_uci: list[str], params_b_uci: list[str]) -> MatchResult:
        """Simulates a match between two parameter sets."""
        # 1. Parse UCI parameters to get current values
        params_a_dict = self._parse_uci_params(params_a_uci)
        params_b_dict = self._parse_uci_params(params_b_uci)

        # 2. Calculate goodness score for each parameter set
        score_a = self._calculate_score(params_a_dict)
        score_b = self._calculate_score(params_b_dict)

        # 3. Calculate Elo difference (A vs B)
        elo_diff = self._get_elo_diff(score_a, score_b)

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

        # Format result like CutechessMan (wins for A, losses for A, draws)
        # Loss for A = wins for B
        result = MatchResult(w=wins_a, l=wins_b, d=draws, elo_diff=elo_diff)

        # print(f"Simulated Run: ScoreA={score_a:.3f}, ScoreB={score_b:.3f}, EloDiff={elo_diff:.2f} -> W={result.w}, L={result.l}, D={result.d}")

        return result

    def get_cutechess_cmd(self, params_a: list[str], params_b: list[str]) -> str:
        """Returns a dummy command string, as no external process is run."""
        return f"FAKE_SIMULATION engine={self.engine_name} games={self.games} params_a='{' '.join(params_a)}' params_b='{' '.join(params_b)}'"

# --- END OF FILE fake_cutechess.py ---
