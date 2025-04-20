# --- START OF FILE main.py ---

import dataclasses
import json
import pathlib
import time
import random
import argparse
import copy
import math # Added
from typing import Optional
# --- Required for artificial params ---
import numpy as np
from scipy.stats import lognorm
# ------------------------------------

from graph import Graph
from spsa import Param, SpsaParams, SpsaTuner
from cutechess import CutechessMan, MatchResult
from fake_cutechess import FakeCutechessMan, FakeSimConfig

# --- Function definitions (cutechess_from_config, spsa_from_config etc.) ---
# --- These are less relevant when using artificial params but kept for normal mode ---

def cutechess_from_config(config_path: str) -> CutechessMan:
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
        return CutechessMan(**config)
    except FileNotFoundError:
        print(f"Warning: Cutechess config file not found at {config_path}. Using defaults for non-simulation runs.")
        return CutechessMan(engine="unknown_engine", book="unknown_book.epd") # Provide defaults
    except json.JSONDecodeError:
         print(f"Warning: Error decoding JSON from {config_path}. Using defaults.")
         return CutechessMan(engine="unknown_engine", book="unknown_book.epd")


def params_from_config(config_path: str) -> list[Param]:
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
        params = []
        for name, cfg in config.items():
            p = Param(name, **cfg)
            # Ensure start_val is set if loading from config where it might be missing
            if getattr(p, 'start_val', None) is None:
                 p.start_val = p.value
            params.append(p)
        return params
    except FileNotFoundError:
         print(f"Error: Parameter config file not found at {config_path}. Cannot proceed without parameters.")
         exit(1)
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {config_path}.")
         exit(1)
    except Exception as e:
        print(f"Error loading parameters from {config_path}: {e}")
        exit(1)


def spsa_from_config(config_path: str):
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
        return SpsaParams(**config)
    except FileNotFoundError:
        print(f"Warning: SPSA config file not found at {config_path}. Using default SPSA parameters.")
        # Provide some defaults if file missing
        return SpsaParams(a=1.0, c=1.0, A=10000, alpha=0.602, gamma=0.101)
    except json.JSONDecodeError:
         print(f"Warning: Error decoding JSON from {config_path}. Using default SPSA parameters.")
         return SpsaParams(a=1.0, c=1.0, A=10000, alpha=0.602, gamma=0.101)


def save_state(spsa: SpsaTuner, state_file_path: str = "./tuner/state.json", is_artificial: bool = False):
    # If using artificial parameters, saving state might be less meaningful
    # unless we also save how they were generated (seed, num_params, etc.).
    # For now, let's allow saving, assuming the run can be recreated with the same seed.
    print(f"Saving state to {state_file_path}...")
    pathlib.Path(state_file_path).parent.mkdir(parents=True, exist_ok=True)

    spsa_params = spsa.spsa
    current_params = spsa.params
    t = spsa.t
    with open(state_file_path, "w") as save_file:
        spsa_params_dict = dataclasses.asdict(spsa_params)
        uci_params_dict = [dataclasses.asdict(p) for p in current_params]

        save_data = {"t": t, "spsa_params": spsa_params_dict,
                     "uci_params": uci_params_dict,
                     # Add flag to indicate if state is from an artificial run
                     "is_artificial": is_artificial}
        # If artificial, maybe save the seed used? Assumes seed is available here.
        # if is_artificial and hasattr(spsa, 'generation_seed'):
        #     save_data['generation_seed'] = spsa.generation_seed

        json.dump(save_data, save_file, indent=4)
    print("State saved.")


def load_state(state_file_path: str, expect_artificial: bool = False) -> tuple[int, SpsaParams, list[Param]] | None:
    """Loads state. If expect_artificial is True, only loads if the saved state matches."""
    state_path = pathlib.Path(state_file_path)
    if not state_path.is_file():
        return None

    print(f"Attempting to load state from {state_file_path}...")
    try:
        with open(state_path) as state:
            state_dict = json.load(state)

        saved_is_artificial = state_dict.get("is_artificial", False)
        if expect_artificial != saved_is_artificial:
             print(f"State file 'is_artificial' flag ({saved_is_artificial}) does not match expectation ({expect_artificial}). Ignoring state file.")
             return None

        params = []
        for cfg in state_dict["uci_params"]:
            # Ensure all necessary fields are present
            required_keys = {"name", "value", "min_value", "max_value", "step", "start_val"}
            if not required_keys.issubset(cfg.keys()):
                 print(f"Warning: State file parameter entry missing required keys: {cfg.get('name', 'Unknown')}. Skipping state load.")
                 return None
            p = Param(**{k: cfg[k] for k in required_keys}) # Explicitly use keys
            params.append(p)

        spsa_params = SpsaParams(**state_dict["spsa_params"])
        t = state_dict["t"]
        print(f"State loaded successfully (t={t}, is_artificial={saved_is_artificial}).")
        return t, spsa_params, params
    except Exception as e:
        print(f"Error loading state from {state_file_path}: {e}. Starting fresh.")
        return None

# --- Function to generate artificial parameters ---
def generate_artificial_params(num_params: int,
                               min_val: float,
                               max_val: float,
                               seed: Optional[int]) -> list[Param]:
    """Generates a list of artificial Param objects."""
    print(f"Generating {num_params} artificial parameters with seed {seed}...")
    params = []
    # Use numpy's random generator for reproducibility with scipy
    rng = np.random.default_rng(seed)

    # Log-normal distribution parameters (shape=s)
    lognorm_shape = 2.583
    lognorm_loc = 0.0
    lognorm_scale = 43.6152
    min_step_value = 1

    for i in range(num_params):
        name = f"ArtParam_{i}"
        value = 0.0 # Initial value is always 0
        start_val = 0.0

        # Generate step size from log-normal distribution
        step = max(min_step_value, lognorm.rvs(s=lognorm_shape, loc=lognorm_loc, scale=lognorm_scale,
                                              size=1, random_state=rng)[0])

        # Create Param object
        p = Param(name=name, value=value, min_value=min_val, max_value=max_val,
                  step=step, start_val=start_val)
        params.append(p)
        # print(f"  Generated: {p}") # Optional: print generated params

    print("Artificial parameter generation complete.")
    return params

# --- Function to create simulation config (modified for artificial params) ---
def create_simulation_config(initial_params: list[Param],
                             base_elo: float, draw_rate: float, draw_sens: float, # Added base config params
                             seed: Optional[int]) -> FakeSimConfig:
    """Creates a simulation configuration, generating optimal offsets and influences."""
    print(f"Generating simulation config with seed {seed}...")
    num_params = len(initial_params)
    rng = np.random.default_rng(seed) # Use separate seed state for sim config generation

    # Log-normal distribution for optimal offsets magnitude
    lognorm_shape = 2.583
    lognorm_loc = 0.0
    lognorm_scale = 43.6152

    optimal_offsets = []
    parameter_influences = []

    # Generate optimal offset magnitudes
    offset_magnitudes = lognorm.rvs(s=lognorm_shape, loc=lognorm_loc, scale=lognorm_scale,
                                    size=num_params, random_state=rng)

    # Generate random signs (-1 or 1) for ~half the offsets
    signs = np.ones(num_params)
    indices_to_flip = rng.choice(num_params, size=num_params // 2, replace=False)
    signs[indices_to_flip] = -1
    rng.shuffle(signs) # Shuffle signs randomly

    print("Generating Optimal Offsets and Influences:")
    for i, param in enumerate(initial_params):
        # Calculate optimal offset
        offset = signs[i] * offset_magnitudes[i]

        # Clamp the *target optimal value* (start_val + offset) within param bounds
        # Since start_val is 0 for artificial params, clamp offset directly
        clamped_offset = max(param.min_value - param.start_val,
                             min(param.max_value - param.start_val, offset))

        if abs(clamped_offset - offset) > 1e-6:
             print(f"  Param '{param.name}': Original offset {offset:.3f} CLAMPED to {clamped_offset:.3f} (Bounds [{param.min_value}, {param.max_value}])")
        optimal_offsets.append(clamped_offset)

        # Generate influence
        influence = rng.uniform(0.0, 1.0)
        parameter_influences.append(influence)
        print(f"  Param '{param.name}': Optimal Offset={clamped_offset:+.3f}, Influence={influence:.3f}")


    config = FakeSimConfig(
        optimal_offsets=optimal_offsets,
        parameter_influences=parameter_influences,
        base_elo_advantage=base_elo, # Use arg value
        base_draw_rate=draw_rate,       # Use arg value
        draw_rate_elo_sensitivity=draw_sens # Use arg value
    )
    # Note: calculate_scaling is called inside FakeCutechessMan init
    print("Simulation config generation complete.")
    return config

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="SPSA Tuner for UCI Chess Engines")
    # Simulation modes
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode using FakeCutechessMan.")
    parser.add_argument("--num-artificial-params", type=int, default=0,
                        help="If > 0 and --simulate is set, generate this many artificial parameters instead of using config.json.")
    parser.add_argument("--artificial-min", type=float, default=-1000.0,
                        help="Min bound for artificial parameters.")
    parser.add_argument("--artificial-max", type=float, default=1000.0,
                        help="Max bound for artificial parameters.")
    parser.add_argument("--sim-seed", type=int, default=None,
                        help="Random seed for simulation parameter generation AND game simulation.")
    parser.add_argument("--sim-base-elo", type=float, default=20.0,
                        help="Target Elo difference between initial and optimal params in simulation.")
    parser.add_argument("--sim-draw-rate", type=float, default=0.60,
                        help="Base draw rate for simulation.")
    parser.add_argument("--sim-draw-elo-sens", type=float, default=0.01,
                        help="Draw rate Elo sensitivity for simulation.")

    # Config files (used if not using artificial params)
    parser.add_argument("--config-params", default="config.json",
                        help="Path to UCI parameters config file (used if not --num-artificial-params).")
    parser.add_argument("--config-spsa", default="spsa.json",
                        help="Path to SPSA parameters config file.")
    parser.add_argument("--config-cutechess", default="cutechess.json",
                        help="Path to Cutechess/Fastchess config file (used for runner settings).")

    # State and output
    parser.add_argument("--state-file", default="./tuner/state.json",
                        help="Path to save/load tuner state.")
    parser.add_argument("--graph-file", default="graph.png",
                        help="Filename for the output graph (in tuner directory).")
    parser.add_argument("--no-state", action="store_true",
                        help="Disable loading and saving of state file.")

    args = parser.parse_args()

    # --- Determine run mode ---
    is_simulating = args.simulate
    use_artificial_params = is_simulating and args.num_artificial_params > 0

    # --- Seed random generators ---
    # Use the same seed for everything if provided, for full reproducibility
    if args.sim_seed is not None:
        random.seed(args.sim_seed)
        np.random.seed(args.sim_seed) # Seed numpy as well for scipy/numpy operations

    # --- Load SPSA parameters ---
    spsa_params_config = spsa_from_config(args.config_spsa)

    # --- Load or Generate UCI parameters ---
    t = 0
    spsa_params = spsa_params_config
    current_params = None
    initial_params_for_sim = None # Store the definitive initial state for simulation

    if use_artificial_params:
        print(f"Generating {args.num_artificial_params} artificial parameters...")
        # Always start fresh (t=0) when generating artificial parameters
        current_params = generate_artificial_params(
            num_params=args.num_artificial_params,
            min_val=args.artificial_min,
            max_val=args.artificial_max,
            seed=args.sim_seed + 1 # Use the seed for generation
        )
        initial_params_for_sim = copy.deepcopy(current_params)
        # Override SPSA A parameter based on expected iterations? Optional.
        # A = max_iterations / 10. Need max_iterations.
        # spsa_params.A = ?

    else:
        # Try loading state (only if not --no-state)
        loaded_state = None
        if not args.no_state:
            # Expect artificial=False when loading state for real/config-based runs
            loaded_state = load_state(args.state_file, expect_artificial=False)

        if loaded_state:
            t, spsa_params, current_params = loaded_state
            print("Resuming from saved state.")
            # Need to reconstruct the initial state if simulating from config
            # This requires the original config file.
            if is_simulating:
                print(f"Loading initial parameters from {args.config_params} for simulation reference...")
                initial_params_def = params_from_config(args.config_params)
                initial_params_for_sim = copy.deepcopy(initial_params_def)
                # Check consistency
                if len(current_params) != len(initial_params_for_sim):
                     print("Error: Loaded state param count mismatch with config file. Cannot resume simulation.")
                     exit(1)
                # Ensure start_val in loaded params matches initial definition
                for i, p_loaded in enumerate(current_params):
                     p_initial = initial_params_for_sim[i]
                     if p_loaded.name != p_initial.name:
                          print(f"Error: Parameter name mismatch between loaded state ({p_loaded.name}) and config ({p_initial.name}).")
                          exit(1)
                     # Use the start_val from the loaded state, assuming it's correct
                     initial_params_for_sim[i].start_val = p_loaded.start_val
            else:
                # Not simulating, initial state less critical but good practice
                initial_params_for_sim = copy.deepcopy(current_params)
                # Restore start_val from loaded state itself
                for p in initial_params_for_sim:
                     pass # start_val should already be loaded correctly by load_state

        else:
            print("Starting from initial configuration.")
            t = 0
            spsa_params = spsa_params_config
            initial_params_def = params_from_config(args.config_params)
            current_params = copy.deepcopy(initial_params_def)
            initial_params_for_sim = copy.deepcopy(initial_params_def)


    # --- Initialize Runner (Cutechess or FakeCutechess) ---
    # Load cutechess config for runner settings like 'games', 'threads' etc.
    cc_config = {}
    try:
        with open(args.config_cutechess) as cc_file:
            cc_config = json.load(cc_file)
    except FileNotFoundError:
        print(f"Warning: Cutechess config '{args.config_cutechess}' not found. Using default runner settings.")
    except json.JSONDecodeError:
        print(f"Warning: Error decoding '{args.config_cutechess}'. Using default runner settings.")

    games_per_iter = cc_config.get("games", 32) # Default games per iteration
    save_rate_iters = cc_config.get("save_rate", 10) # Default save rate

    if is_simulating:
        print("Initializing SIMULATION runner...")
        if initial_params_for_sim is None:
             print("Error: Initial parameters for simulation not set.")
             exit(1)
        # Create the simulation configuration
        sim_config = create_simulation_config(
             initial_params=initial_params_for_sim,
             base_elo=args.sim_base_elo,
             draw_rate=args.sim_draw_rate,
             draw_sens=args.sim_draw_elo_sens,
             seed=args.sim_seed + 2 # Use same seed for consistency
        )
        runner = FakeCutechessMan(
            initial_params=initial_params_for_sim,
            sim_config=sim_config,
            games=games_per_iter,
            # Pass other params from cutechess.json if needed
            engine_name=cc_config.get("engine", "fake_engine"),
            book=cc_config.get("book", "fake_book.epd"),
            tc=cc_config.get("tc", 5.0),
            hash_size=cc_config.get("hash", 8),
            threads=cc_config.get("threads", 1)
        )
    else:
        print("Initializing LIVE runner (Cutechess/Fastchess)...")
        runner = CutechessMan(**cc_config) # Pass full config read from file
        games_per_iter = runner.games # Get actual games from live runner
        save_rate_iters = runner.save_rate # Get actual save rate


    # --- Initialize SPSA Tuner ---
    spsa = SpsaTuner(spsa_params, current_params, runner)
    spsa.t = t # Restore time step (games played)
    # Store initial params reference if needed (e.g., for simulation checks)
    spsa.initial_params = initial_params_for_sim
    # Optionally store seed if needed for saving state:
    # spsa.generation_seed = args.sim_seed if use_artificial_params else None


    # --- Initialize Graph ---
    graph_path = pathlib.Path("./tuner/") / args.graph_file
    graph = Graph()

    # --- Tuning Loop ---
    start_t = spsa.t # Games played before this run starts
    loop_start_time = time.time()

    print("\n--- Starting Tuning ---")
    print("Initial state:")
    for param in spsa.params:
        print(f"  {param}")
    print(f"Runner: {'FakeCutechessMan (Simulation)' if is_simulating else 'CutechessMan (Live)'}")
    print(f"Games per iteration: {games_per_iter}")
    print("-" * 20)

    try:
        while True: # Add a condition to stop? (e.g., max iterations/games)
            iter_start_time = time.time()

            # Check for potential division by zero in spsa calculation if t=0 initially
            if spsa.t == 0 and spsa.spsa.gamma > 0:
                # Maybe run one initial evaluation step without update? Or adjust SPSA formula?
                # Simplest: SpsaTuner internally starts k from t+1, avoiding division by t^gamma when t=0.
                 pass # Should be handled by k = t + 1 in SPSA step

            spsa.step() # Perform one SPSA step (runs games_per_iter games)
            iter_time = time.time() - iter_start_time

            current_run_games = spsa.t - start_t
            if current_run_games > 0:
                 current_run_time = time.time() - loop_start_time
                 avg_time_per_game = current_run_time / current_run_games
                 avg_time_per_iter = avg_time_per_game * games_per_iter
            else:
                 avg_time_per_game = 0
                 avg_time_per_iter = 0

            # Graph update
            # Make sure graph directory exists
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph.update(spsa.t, copy.deepcopy(spsa.params))
            graph.save(graph_path.name)

            current_iter = int((spsa.t + games_per_iter -1) / games_per_iter) # Iteration number (starts at 1)

            # Save state based on iterations (if not disabled)
            if not args.no_state and current_iter > 0 and current_iter % save_rate_iters == 0:
                save_state(spsa, args.state_file, is_artificial=use_artificial_params)

            print(f"\nIteration {current_iter} (Game {spsa.t}) completed.")
            print(f"  Time: {iter_time:.2f}s ({avg_time_per_iter:.2f}s avg/iter, {avg_time_per_game:.3f}s avg/game)")
            print("  Current Parameters:")
            for param in spsa.params:
                 print(f"    {param}")
            if is_simulating:
                 # Optionally print distance to optimum for simulation runs
                 print_distance_to_optimum(spsa.params, runner.sim_config.optimal_offsets)

            print("-" * 20)

            # Optional: Add stopping condition (e.g., max iterations)
            # if current_iter >= MAX_ITERATIONS:
            #     break

    except KeyboardInterrupt:
         print("\nInterrupted by user.")
    except Exception as e:
         print(f"\nAn error occurred during tuning: {e}")
         import traceback
         traceback.print_exc() # Print stack trace for debugging
    finally:
        print("\n--- Tuning Finished ---")
        if not args.no_state:
            print("Saving final state...")
            save_state(spsa, args.state_file, is_artificial=use_artificial_params)

        current_run_games = spsa.t - start_t
        if current_run_games > 0:
             current_run_time = time.time() - loop_start_time
             avg_time_per_game = current_run_time / current_run_games
             avg_time_per_iter = avg_time_per_game * games_per_iter
        else:
            avg_time_per_game = 0
            avg_time_per_iter = 0
        current_iter = int((spsa.t + games_per_iter -1) / games_per_iter)

        print("\nFinal results:")
        print(f"  Total iterations: {current_iter}")
        print(f"  Total games: {spsa.t}")
        print(f"  Avg time/iter: {avg_time_per_iter:.2f}s")
        print(f"  Avg time/game: {avg_time_per_game:.3f}s")
        print("  Final parameters:")
        for param in spsa.params:
            print(f"    {param}")

        # If simulating, print final distance to optimal values
        if is_simulating and hasattr(runner, 'sim_config'):
            print("\n--- Simulation Performance ---")
            print("Target Optimal Offsets vs Final Offsets:")
            total_weighted_dev = 0
            if hasattr(runner, '_calculate_total_weighted_deviation'):
                 # Recalculate deviation using the runner's internal method if possible
                 final_params_dict = runner._parse_uci_params([p.as_uci for p in spsa.params])
                 total_weighted_dev = runner._calculate_total_weighted_deviation(final_params_dict)
                 print(f"  Final Total Weighted Deviation: {total_weighted_dev:.4f}")
                 if runner.sim_config.elo_per_deviation_unit != 0:
                      final_elo_vs_initial = (runner.sim_config.initial_total_weighted_deviation - total_weighted_dev) * runner.sim_config.elo_per_deviation_unit
                      print(f"  Estimated Elo vs Initial Params: {final_elo_vs_initial:+.2f}")
                 else:
                      print(f"  Estimated Elo vs Initial Params: N/A (Initial was optimal)")

            print_distance_to_optimum(spsa.params, runner.sim_config.optimal_offsets)


def print_distance_to_optimum(final_params: list[Param], optimal_offsets: list[float]):
    """Helper function to calculate distance metrics."""
    total_dist = 0
    max_dist = 0
    if len(final_params) != len(optimal_offsets):
        print("Warning: Mismatch between final params and optimal offsets count.")
        return -1.0, -1.0

    print(f"  Parameters:")
    for i, param in enumerate(final_params):
        final_offset = param.value - param.start_val # Calculate offset from its start_val
        optimal_offset = optimal_offsets[i]
        diff = abs(final_offset - optimal_offset)
        total_dist += diff
        max_dist = max(max_dist, diff)

        print(f"    {param.name}: Dist to Optimum: {diff:.4f}, Step: {param.step:.4f}")

    avg_dist = total_dist / len(final_params) if final_params else 0


    print(f"  Avg Abs Dist to Optimum: {avg_dist:.4f}, Max Abs Dist: {max_dist:.4f}")


if __name__ == "__main__":
    main()

# --- END OF FILE main.py ---
