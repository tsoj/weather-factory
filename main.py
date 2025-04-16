# --- START OF FILE main.py ---

import dataclasses
import json
import pathlib
import time
import random # Added
import argparse # Added
from graph import Graph
from spsa import Param, SpsaParams, SpsaTuner
from cutechess import CutechessMan, MatchResult # MatchResult potentially needed by fake one too
from fake_cutechess import FakeCutechessMan, FakeSimConfig # Added
import copy

# --- Function definitions (cutechess_from_config, params_from_config, etc.) remain the same ---

def cutechess_from_config(config_path: str) -> CutechessMan:
    with open(config_path) as config_file:
        config = json.load(config_file)
    # Keep the original signature, but the object returned might be FakeCutechessMan
    return CutechessMan(**config)


def params_from_config(config_path: str) -> list[Param]:
    with open(config_path) as config_file:
        config = json.load(config_file)
    # Store the initial value explicitly for the simulation
    params = []
    for name, cfg in config.items():
        p = Param(name, **cfg)
        p.start_val = p.value # Explicitly store start value
        params.append(p)
    return params


def spsa_from_config(config_path: str):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return SpsaParams(**config)


def save_state(spsa: SpsaTuner, state_file_path: str = "./tuner/state.json"):
    # Make sure directories exist
    pathlib.Path(state_file_path).parent.mkdir(parents=True, exist_ok=True)

    spsa_params = spsa.spsa
    # Use spsa.initial_params to save the state consistent with simulation start
    # Or better, save the *current* params derived from initial + spsa steps
    current_params = spsa.params # Get current parameter state
    t = spsa.t
    with open(state_file_path, "w") as save_file:
        spsa_params_dict = dataclasses.asdict(spsa_params)
        # Save current params, including their original start_val for consistency
        uci_params_dict = []
        for p in current_params:
            p_dict = dataclasses.asdict(p)
            # Ensure start_val is saved if it exists, otherwise add it from initial if needed
            if not hasattr(p, 'start_val') and hasattr(spsa, 'initial_params'):
                 initial_p = next((ip for ip in spsa.initial_params if ip.name == p.name), None)
                 if initial_p:
                     p_dict['start_val'] = initial_p.start_val
            elif not hasattr(p, 'start_val'):
                 # Fallback if initial_params isn't available (shouldn't happen with proper init)
                 p_dict['start_val'] = p.value # Best guess if missing
            uci_params_dict.append(p_dict)


        json.dump({"t": t, "spsa_params": spsa_params_dict,
                  "uci_params": uci_params_dict}, save_file, indent=4) # Added indent


def load_state(state_file_path: str) -> tuple[int, SpsaParams, list[Param]] | None:
    state_path = pathlib.Path(state_file_path)
    if not state_path.is_file():
        return None

    print(f"Loading state from {state_file_path}...")
    with open(state_path) as state:
        state_dict = json.load(state)
        # When loading, reconstruct Param ensuring start_val is handled
        params = []
        for cfg in state_dict["uci_params"]:
            # Check if start_val is in the saved state, otherwise default might be needed
            # The Param dataclass now handles start_val in post_init, but explicit loading is safer
            p = Param(name=cfg["name"], value=cfg["value"], min_value=cfg["min_value"],
                      max_value=cfg["max_value"], step=cfg["step"])
            # Explicitly set start_val from saved state if available
            if "start_val" in cfg:
                 p.start_val = cfg["start_val"]
            else:
                 # If missing from state file (older format?), assume current value was start
                 # Or better, require it / try reloading initial config (complex).
                 # Let's assume it *should* be there if saved correctly.
                 print(f"Warning: 'start_val' missing for param {p.name} in state file. Using current value as start.")
                 p.start_val = cfg["value"]
            params.append(p)

        spsa_params = SpsaParams(**state_dict["spsa_params"])
        t = state_dict["t"]
        print("State loaded.")
        return t, spsa_params, params


def create_simulation_config(initial_params: list[Param], seed: int | None = None) -> FakeSimConfig:
    """Creates a random simulation configuration."""
    if seed is not None:
        random.seed(seed)

    num_params = len(initial_params)
    optimal_offsets = []
    parameter_influences = []

    print("Generating Fake Simulation Parameters...")
    for param in initial_params:
        # Generate optimal offset: somewhere within +/- 50% of the range from the start value?
        range_size = param.max_value - param.min_value
        # Ensure start_val exists
        start_val = getattr(param, 'start_val', param.value)
        max_offset_from_start = range_size * 0.5
        # Calculate potential min/max offsets based on bounds relative to start_val
        min_possible_offset = param.min_value - start_val
        max_possible_offset = param.max_value - start_val

        # Generate random offset within a fraction of the range, respecting boundaries
        offset_magnitude = random.uniform(0, max_offset_from_start)
        offset_sign = random.choice([-1, 1])
        potential_offset = offset_sign * offset_magnitude

        # Clamp offset to ensure optimal value is within [min_value, max_value]
        optimal_offset = max(min_possible_offset, min(max_possible_offset, potential_offset))
        optimal_offsets.append(optimal_offset)

        # Generate influence: random value between 0.1 and 1.0?
        influence = random.uniform(0.1, 1.0)
        parameter_influences.append(influence)
        print(f"  Param '{param.name}': Optimal Offset={optimal_offset:+.3f}, Influence={influence:.3f}")

    # You might want to tune these defaults
    config = FakeSimConfig(
        optimal_offsets=optimal_offsets,
        parameter_influences=parameter_influences,
        base_elo_advantage=random.uniform(15, 40), # Randomize base strength difference
        base_draw_rate=random.uniform(0.4, 0.7),    # Randomize typical draw rate
        draw_rate_elo_sensitivity=random.uniform(0.005, 0.015) # Randomize sensitivity
    )
    print("Simulation generation complete.")
    return config


def main():
    parser = argparse.ArgumentParser(description="SPSA Tuner for UCI Chess Engines")
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode using FakeCutechessMan.")
    parser.add_argument("--config-params", default="config.json",
                        help="Path to UCI parameters config file.")
    parser.add_argument("--config-spsa", default="spsa.json",
                        help="Path to SPSA parameters config file.")
    parser.add_argument("--config-cutechess", default="cutechess.json",
                        help="Path to Cutechess/Fastchess config file.")
    parser.add_argument("--state-file", default="./tuner/state.json",
                        help="Path to save/load tuner state.")
    parser.add_argument("--graph-file", default="graph.png",
                        help="Filename for the output graph (in tuner directory).")
    parser.add_argument("--sim-seed", type=int, default=None,
                        help="Random seed for simulation parameter generation.")
    args = parser.parse_args()

    # --- Load base configurations ---
    spsa_params_config = spsa_from_config(args.config_spsa)
    # Load initial parameters definition
    initial_params_def = params_from_config(args.config_params)

    # --- Load state or use initial config ---
    loaded_state = load_state(args.state_file)
    if loaded_state:
        t, spsa_params, current_params = loaded_state
        print("Resuming from saved state.")
        # We need the *initial* parameters for the simulation base,
        # ensure the loaded Param objects have `start_val` correctly set.
        # The load_state function should handle this.
        initial_params_for_sim = copy.deepcopy(current_params)
        # Ensure start_val is correctly set from loaded state
        for p in initial_params_for_sim:
             if not hasattr(p, 'start_val'):
                  # This ideally shouldn't happen if save/load is correct
                  print(f"Error: Loaded parameter {p.name} lacks start_val. Cannot guarantee simulation consistency.")
                  # As a fallback, maybe use the value from initial_params_def?
                  initial_def = next((ipd for ipd in initial_params_def if ipd.name == p.name), None)
                  p.start_val = initial_def.start_val if initial_def and hasattr(initial_def, 'start_val') else p.value


    else:
        print("No saved state found or error loading. Starting from initial configuration.")
        t = 0
        spsa_params = spsa_params_config
        current_params = copy.deepcopy(initial_params_def) # Start with initial definition
        # Ensure start_val is set for the initial run
        for p in current_params:
            if not hasattr(p, 'start_val'):
                p.start_val = p.value
        initial_params_for_sim = copy.deepcopy(current_params)


    # --- Initialize Cutechess or FakeCutechess ---
    if args.simulate:
        print("Running in SIMULATION mode.")
        # Create simulation config based on *initial* parameters
        sim_config = create_simulation_config(initial_params_for_sim, seed=args.sim_seed)
        # Load cutechess config just to get 'games', 'tc' etc.
        with open(args.config_cutechess) as cc_config_file:
            cc_config = json.load(cc_config_file)
        runner = FakeCutechessMan(
            initial_params=initial_params_for_sim, # Pass initial params for reference
            sim_config=sim_config,
            games=cc_config.get("games", 32),
            engine_name=cc_config.get("engine", "fake_engine"),
             # Pass other params from cutechess.json if needed by FakeCutechessMan structure
            book=cc_config.get("book", "fake_book.epd"),
            tc=cc_config.get("tc", 5.0),
            hash_size=cc_config.get("hash", 8),
            threads=cc_config.get("threads", 1)
        )
        games_per_iter = runner.games # Get from the instance
        save_rate_iters = cc_config.get("save_rate", 10) # Base save rate on iterations

    else:
        print("Running in LIVE mode (using Cutechess/Fastchess).")
        runner = cutechess_from_config(args.config_cutechess)
        games_per_iter = runner.games # Get from the instance
        save_rate_iters = runner.save_rate # Use save_rate from config directly

    # --- Initialize SPSA Tuner ---
    # Pass the *current* parameters (loaded or initial) to SpsaTuner
    spsa = SpsaTuner(spsa_params, current_params, runner)
    spsa.t = t # Restore time step

    # Store initial params separately if needed (e.g., for reference or simulation checks)
    spsa.initial_params = initial_params_for_sim


    # --- Initialize Graph ---
    graph_path = pathlib.Path("./tuner/") / args.graph_file
    graph = Graph()
    # If resuming, might want to load previous graph history (more complex)

    # --- Tuning Loop ---
    avg_time = 0
    start_t = spsa.t # Games played before this run
    loop_start_time = time.time()

    print("Initial state: ")
    for param in spsa.params:
        print(param)
    print()

    try:
        while True: # Add a condition to stop? e.g., max iterations
            iter_start_time = time.time()
            spsa.step() # This now uses either the real or fake runner internally
            iter_time = time.time() - iter_start_time
            # avg_time calculation needs adjustment if resuming
            # Total time elapsed in this execution / number of games *in this execution*
            current_run_games = spsa.t - start_t
            if current_run_games > 0:
                 current_run_time = time.time() - loop_start_time
                 avg_time_per_game = current_run_time / current_run_games
                 avg_time_per_iter = avg_time_per_game * games_per_iter
            else:
                 avg_time_per_game = 0
                 avg_time_per_iter = 0


            graph.update(spsa.t, copy.deepcopy(spsa.params))
            graph.save(graph_path.name) # Save with the specified filename

            current_iter = int(spsa.t / games_per_iter) # Iteration number

            # Save state based on iterations
            if current_iter > 0 and current_iter % save_rate_iters == 0:
                print(f"Saving state at iteration {current_iter} (game {spsa.t})...")
                save_state(spsa, args.state_file)

            print(
                f"iterations: {current_iter} ({avg_time_per_iter:.2f}s per iter)")
            print(
                f"games: {spsa.t} ({avg_time_per_game:.2f}s per game)")
            for param in spsa.params:
                print(param)
            print("-" * 20) # Separator

            # Optional: Add a stopping condition
            # if current_iter >= max_iterations:
            #     break

    except KeyboardInterrupt:
         print("\nInterrupted by user.")
    finally:
        print("Saving final state...")
        save_state(spsa, args.state_file)
        print("Final results: ")
        current_run_games = spsa.t - start_t
        if current_run_games > 0:
             current_run_time = time.time() - loop_start_time
             avg_time_per_game = current_run_time / current_run_games
             avg_time_per_iter = avg_time_per_game * games_per_iter
        else:
            avg_time_per_game = 0
            avg_time_per_iter = 0
        current_iter = int(spsa.t / games_per_iter)
        print(
            f"iterations: {current_iter} ({avg_time_per_iter:.2f}s per iter)")
        print(
            f"games: {spsa.t} ({avg_time_per_game:.2f}s per game)")
        print("Final parameters: ")
        for param in spsa.params:
            print(param)

        # If simulating, print how close we got to the optimal values
        if args.simulate and hasattr(runner, 'sim_config'):
            print("\n--- Simulation Performance ---")
            print("Target Optimal Offsets vs Final Offsets:")
            total_dist = 0
            max_dist = 0
            for i, param in enumerate(spsa.params):
                initial_val = spsa.initial_params[i].start_val
                final_offset = param.value - initial_val
                optimal_offset = runner.sim_config.optimal_offsets[i]
                diff = final_offset - optimal_offset
                print(f"  {param.name}: Target={optimal_offset:+.3f}, Final={final_offset:+.3f}, Diff={diff:+.3f}")
                total_dist += abs(diff)
                max_dist = max(max_dist, abs(diff))
            print(f"Average Absolute Difference from Optimum: {total_dist / len(spsa.params):.4f}")
            print(f"Maximum Absolute Difference from Optimum: {max_dist:.4f}")


if __name__ == "__main__":
    main()

# --- END OF FILE main.py ---
