#!/usr/bin/env python3
"""
SPSA parameter tuner for UCI chess engines with simulation capabilities.
"""

import argparse
import copy
import dataclasses
import json
import pathlib
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.stats import lognorm

from spsa import Param, SpsaParams, SpsaTuner
from cutechess import CutechessMan, MatchResult
from fake_cutechess import FakeCutechessMan, FakeSimConfig, calculate_total_weighted_deviation
from param_gen import ParamGenerator
from graph import Graph


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SPSA Tuner for UCI Chess Engines")

    # Simulation mode
    sim_group = parser.add_argument_group('Simulation')
    sim_group.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode using FakeCutechessMan")
    sim_group.add_argument("--num-artificial-params", type=int, default=0,
                        help="If > 0 and --simulate is set, generate this many artificial parameters")
    sim_group.add_argument("--artificial-min", type=float, default=-float("inf"),
                        help="Min bound for artificial parameters")
    sim_group.add_argument("--artificial-max", type=float, default=float("inf"),
                        help="Max bound for artificial parameters")
    sim_group.add_argument("--sim-seed", type=int, default=None,
                        help="Random seed for simulation")
    sim_group.add_argument("--rel-step-size", type=float, default=0.1,
                      help="Step size as fraction of parameter value (default: 0.1)")
    sim_group.add_argument("--rel-optimal-change", type=float, default=0.1,
                      help="Optimal value change as fraction of parameter value (default: 0.1)")
    sim_group.add_argument("--sim-base-elo", type=float, default=20.0,
                        help="Target Elo difference between initial and optimal params")
    sim_group.add_argument("--sim-draw-rate", type=float, default=0.60,
                        help="Base draw rate for simulation")
    sim_group.add_argument("--sim-draw-elo-sens", type=float, default=0.01,
                        help="Draw rate Elo sensitivity for simulation")

    # Config files
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument("--config-params", default="config.json",
                        help="Path to UCI parameters config file")
    config_group.add_argument("--config-spsa", default="spsa.json",
                        help="Path to SPSA parameters config file")
    config_group.add_argument("--config-cutechess", default="cutechess.json",
                        help="Path to Cutechess config file")

    # State and output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument("--state-file", default="./tuner/state.json",
                        help="Path to save/load tuner state")
    output_group.add_argument("--graph-file", default="graph.png",
                        help="Filename for the output graph (in tuner directory)")
    output_group.add_argument("--no-state", action="store_true",
                        help="Disable loading and saving of state")
    output_group.add_argument("--max-iterations", type=int, default=0,
                        help="Maximum iterations to run (0 = unlimited)")

    return parser.parse_args()


def load_config(config_path: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load JSON configuration with fallback to default."""
    if default is None:
        default = {}

    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using defaults.")
        return default
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from {config_path}. Using defaults.")
        return default


def cutechess_from_config(config_path: str) -> CutechessMan:
    """Create CutechessMan instance from config file."""
    default_config = {
        "engine": "unknown_engine",
        "book": "unknown_book.epd",
        "games": 32,
        "threads": 1
    }

    config = load_config(config_path, default_config)
    return CutechessMan(**config)


def params_from_config(config_path: str) -> List[Param]:
    """Load parameter definitions from config file."""
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)

        params = []
        for name, cfg in config.items():
            p = Param(name, **cfg)
            # Ensure start_val is set
            if getattr(p, 'start_val', None) is None:
                p.start_val = p.value
            params.append(p)
        return params
    except FileNotFoundError:
        print(f"Error: Parameter config file not found at {config_path}.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}.")
        exit(1)
    except Exception as e:
        print(f"Error loading parameters from {config_path}: {e}")
        exit(1)


def spsa_from_config(config_path: str) -> SpsaParams:
    """Load SPSA configuration from file."""
    default_config = {
        "a": 1.0,
        "c": 1.0,
        "A": 10000,
        "alpha": 0.602,
        "gamma": 0.101
    }

    config = load_config(config_path, default_config)
    return SpsaParams(**config)


def save_state(
    spsa: SpsaTuner,
    state_file_path: str = "./tuner/state.json",
    is_artificial: bool = False,
    sim_config: Optional[Dict[str, Any]] = None
):
    """Save tuner state to file."""
    print(f"Saving state to {state_file_path}...")
    pathlib.Path(state_file_path).parent.mkdir(parents=True, exist_ok=True)

    spsa_params = spsa.spsa
    current_params = spsa.params
    t = spsa.t

    # Convert objects to dictionaries
    spsa_params_dict = dataclasses.asdict(spsa_params)
    uci_params_dict = [dataclasses.asdict(p) for p in current_params]

    save_data = {
        "t": t,
        "spsa_params": spsa_params_dict,
        "uci_params": uci_params_dict,
        "is_artificial": is_artificial
    }

    # Save simulation config if provided
    if is_artificial and sim_config:
        save_data['sim_config'] = sim_config

    with open(state_file_path, "w") as save_file:
        json.dump(save_data, save_file, indent=4)

    print("State saved.")


def load_state(
    state_file_path: str,
    expect_artificial: bool = False
) -> Optional[Tuple[int, SpsaParams, List[Param], Optional[Dict[str, Any]]]]:
    """
    Load tuner state from file.

    Returns:
        Tuple of (t, spsa_params, params, sim_config) or None if loading failed
    """
    state_path = pathlib.Path(state_file_path)
    if not state_path.is_file():
        return None

    print(f"Attempting to load state from {state_file_path}...")
    try:
        with open(state_path) as state_file:
            state_dict = json.load(state_file)

        saved_is_artificial = state_dict.get("is_artificial", False)
        if expect_artificial != saved_is_artificial:
            print(f"State file 'is_artificial' flag ({saved_is_artificial}) "
                  f"does not match expectation ({expect_artificial}). Ignoring state file.")
            return None

        # Load parameters
        params = []
        for cfg in state_dict["uci_params"]:
            required_keys = {"name", "value", "min_value", "max_value", "step", "start_val"}
            if not required_keys.issubset(cfg.keys()):
                print(f"Warning: State file parameter entry missing required keys: "
                      f"{cfg.get('name', 'Unknown')}. Skipping state load.")
                return None

            p = Param(**{k: cfg[k] for k in required_keys})
            params.append(p)

        # Load SPSA configuration
        spsa_params = SpsaParams(**state_dict["spsa_params"])
        t = state_dict["t"]

        # Load simulation config if available
        sim_config = state_dict.get("sim_config")

        print(f"State loaded successfully (t={t}, is_artificial={saved_is_artificial}).")
        return t, spsa_params, params, sim_config

    except Exception as e:
        print(f"Error loading state from {state_file_path}: {e}. Starting fresh.")
        return None


def print_distance_to_optimum(sim_config: FakeSimConfig, params: List[Param]):
    """Print distance metrics between current parameters and optimal values."""
    total_dist = 0
    max_dist = 0

    if len(params) != len(sim_config.optimal_values):
        print("Warning: Mismatch between final params and optimal offsets count.")
        return -1.0, -1.0

    print(f"  Parameters:")
    for i, param in enumerate(params):
        optimal_value = sim_config.optimal_values[i]
        diff = abs(param.value - optimal_value)
        total_dist += diff
        max_dist = max(max_dist, diff)

        print(f"    {param.name}: Dist to Optimum: {diff:.4f}, Step: {param.step:.4f}")

    avg_dist = total_dist / len(params) if params else 0

    deviation = calculate_total_weighted_deviation(sim_config, params)

    print(f"  Avg Abs Dist to Optimum: {avg_dist:.4f}, Max Abs Dist: {max_dist:.4f}, Deviation: {deviation:.4f}")


def main():
    """Main function to run the SPSA tuner."""
    args = parse_arguments()

    # Determine run mode
    is_simulating = args.simulate
    use_artificial_params = is_simulating and args.num_artificial_params > 0

    # Seed random generators
    if args.sim_seed is not None:
        random.seed(args.sim_seed)
        np.random.seed(args.sim_seed)

    # Load SPSA parameters
    spsa_params_config = spsa_from_config(args.config_spsa)

    # Initialize variables
    t = 0
    spsa_params = spsa_params_config
    current_params = None
    initial_params_for_sim = None
    sim_config = None

    # Handle artificial parameters or load from config
    if use_artificial_params:
        print(f"Generating {args.num_artificial_params} artificial parameters...")
        param_gen = ParamGenerator(seed=args.sim_seed)

        # Generate parameters
        current_params = param_gen.generate_parameters(
            num_params=args.num_artificial_params,
            min_val=args.artificial_min,
            max_val=args.artificial_max,
            rel_step_size=args.rel_step_size
        )

        # Keep a copy of initial parameters
        initial_params_for_sim = copy.deepcopy(current_params)

        # Generate optimal values and influence weights
        optimal_values, parameter_influences = param_gen.generate_optimal_values(
            params=current_params,
            rel_change=args.rel_optimal_change
        )

        # Create simulation configuration
        sim_config = {
            "optimal_values": optimal_values,
            "parameter_influences": parameter_influences,
            "base_elo_advantage": args.sim_base_elo,
            "base_draw_rate": args.sim_draw_rate,
            "draw_rate_elo_sensitivity": args.sim_draw_elo_sens
        }

    else:
        # Try loading state if not using artificial parameters
        loaded_state = None
        if not args.no_state:
            loaded_state = load_state(args.state_file, expect_artificial=use_artificial_params)

        if loaded_state:
            t, spsa_params, current_params, saved_sim_config = loaded_state
            print("Resuming from saved state.")

            # Use saved simulation config if available
            if is_simulating and saved_sim_config:
                sim_config = saved_sim_config

            # Load initial parameters for simulation reference if simulating
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
                        print(f"Error: Parameter name mismatch between loaded state ({p_loaded.name}) "
                              f"and config ({p_initial.name}).")
                        exit(1)
                    initial_params_for_sim[i].start_val = p_loaded.start_val
            else:
                initial_params_for_sim = copy.deepcopy(current_params)

        else:
            print("Starting from initial configuration.")
            t = 0
            spsa_params = spsa_params_config
            current_params = params_from_config(args.config_params)
            initial_params_for_sim = copy.deepcopy(current_params)

    # Load cutechess config for runner settings
    cc_config = load_config(args.config_cutechess, default={
        "engine": "unknown_engine",
        "book": "unknown_book.epd",
        "games": 32,
        "threads": 1,
        "save_rate": 10
    })

    games_per_iter = cc_config.get("games", 32)
    save_rate_iters = cc_config.get("save_rate", 10)

    # Initialize runner (Cutechess or FakeCutechess)
    if is_simulating:
        print("Initializing SIMULATION runner...")
        if not sim_config:
            print("Error: Simulation configuration is missing.")
            exit(1)

        runner = FakeCutechessMan(
            initial_params=initial_params_for_sim,
            sim_config=FakeSimConfig(**sim_config),
            games=games_per_iter,
            engine_name=cc_config.get("engine", "fake_engine"),
            book=cc_config.get("book", "fake_book.epd"),
            tc=cc_config.get("tc", 5.0),
            hash_size=cc_config.get("hash", 8),
            threads=cc_config.get("threads", 1)
        )
    else:
        print("Initializing LIVE runner (Cutechess)...")
        runner = cutechess_from_config(args.config_cutechess)
        games_per_iter = runner.games
        save_rate_iters = cc_config.get("save_rate", save_rate_iters)

    # Initialize SPSA Tuner
    spsa = SpsaTuner(spsa_params, current_params, runner)
    spsa.t = t  # Restore time step

    # Initialize Graph
    graph_path = pathlib.Path("./tuner") / args.graph_file
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph = Graph()

    # Tuning Loop
    start_t = spsa.t  # Games played before this run
    loop_start_time = time.time()

    print("\n--- Starting Tuning ---")
    print("Initial state:")
    for param in spsa.params:
        print(f"  {param}")
    print(f"Runner: {'FakeCutechessMan (Simulation)' if is_simulating else 'CutechessMan (Live)'}")
    print(f"Games per iteration: {games_per_iter}")
    print("-" * 20)

    try:
        current_iter = 0
        while True:
            iter_start_time = time.time()

            # Perform SPSA step
            spsa.step()

            iter_time = time.time() - iter_start_time

            # Calculate statistics
            current_run_games = spsa.t - start_t
            if current_run_games > 0:
                current_run_time = time.time() - loop_start_time
                avg_time_per_game = current_run_time / current_run_games
                avg_time_per_iter = avg_time_per_game * games_per_iter
            else:
                avg_time_per_game = 0
                avg_time_per_iter = 0

            # Update and save graph
            graph.update(spsa.t, copy.deepcopy(spsa.params))
            graph.save(str(graph_path))

            # Calculate current iteration
            current_iter = int((spsa.t + games_per_iter - 1) / games_per_iter)

            # Save state based on iterations
            if not args.no_state and current_iter > 0 and current_iter % save_rate_iters == 0:
                save_state(spsa, args.state_file, is_artificial=use_artificial_params, sim_config=sim_config)

            # Print status
            print(f"\nIteration {current_iter} (Game {spsa.t}) completed.")
            print(f"  Time: {iter_time:.2f}s ({avg_time_per_iter:.2f}s avg/iter, "
                  f"{avg_time_per_game:.3f}s avg/game)")
            print("  Current Parameters:")
            for param in spsa.params:
                print(f"    {param}")

            # Print distance to optimum if simulating
            if is_simulating and hasattr(runner, 'sim_config'):
                print_distance_to_optimum(runner.sim_config, spsa.params)

            print("-" * 20)

            # Check for stopping condition
            if args.max_iterations > 0 and current_iter >= args.max_iterations:
                print(f"Maximum iterations ({args.max_iterations}) reached. Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during tuning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Tuning Finished ---")

        # Save final state
        if not args.no_state:
            save_state(spsa, args.state_file, is_artificial=use_artificial_params, sim_config=sim_config)

        # Calculate final statistics
        current_run_games = spsa.t - start_t
        if current_run_games > 0:
            current_run_time = time.time() - loop_start_time
            avg_time_per_game = current_run_time / current_run_games
            avg_time_per_iter = avg_time_per_game * games_per_iter
        else:
            avg_time_per_game = 0
            avg_time_per_iter = 0

        print("\nFinal results:")
        print(f"  Total iterations: {current_iter}")
        print(f"  Total games: {spsa.t}")
        print(f"  Avg time/iter: {avg_time_per_iter:.2f}s")
        print(f"  Avg time/game: {avg_time_per_game:.3f}s")
        print("  Final parameters:")
        for param in spsa.params:
            print(f"    {param}")

        # Print simulation performance if applicable
        if is_simulating and hasattr(runner, 'sim_config'):
            print("\n--- Simulation Performance ---")

            # Calculate deviation if possible
            total_weighted_dev = 0
            if hasattr(runner, '_calculate_total_weighted_deviation'):
                final_params_dict = runner._parse_uci_params([p.as_uci for p in spsa.params])
                total_weighted_dev = runner._calculate_total_weighted_deviation(final_params_dict)
                print(f"  Final Total Weighted Deviation: {total_weighted_dev:.4f}")

                if runner.sim_config.elo_per_deviation_unit != 0:
                    final_elo_vs_initial = (runner.sim_config.initial_total_weighted_deviation - total_weighted_dev) * runner.sim_config.elo_per_deviation_unit
                    print(f"  Estimated Elo vs Initial Params: {final_elo_vs_initial:+.2f}")
                else:
                    print(f"  Estimated Elo vs Initial Params: N/A (Initial was optimal)")

            print_distance_to_optimum(runner.sim_config, spsa.params)


if __name__ == "__main__":
    main()
