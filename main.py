
import dataclasses
import json
import pathlib
import time
from graph import Graph
from spsa import Param, SpsaParams, SpsaTuner
from cuteataxx import CuteataxxMan
import copy


def cuteataxx_from_config(config_path: str) -> CuteataxxMan:
    with open(config_path) as config_file:
        config = json.load(config_file)
    return CuteataxxMan(**config)


def params_from_config(config_path: str) -> list[Param]:
    with open(config_path) as config_file:
        config = json.load(config_file)
    return [Param(name, **cfg) for name, cfg in config.items()]


def spsa_from_config(config_path: str):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return SpsaParams(**config)


def save_state(spsa: SpsaTuner):
    save_file = "./tuner/state.json"
    spsa_params = spsa.spsa
    uai_params = spsa.uai_params
    t = spsa.t
    with open(save_file, "w") as save_file:
        spsa_params = dataclasses.asdict(spsa_params)
        uai_params = [dataclasses.asdict(
            uai_param) for uai_param in uai_params]

        json.dump({"t": t, "spsa_params": spsa_params,
                  "uai_params": uai_params}, save_file)


def main():

    state_path = pathlib.Path("./tuner/state.json")
    t = 0
    if state_path.is_file():
        print("hey")
        with open(state_path) as state:
            state_dict = json.load(state)
            params = [Param(cfg["name"], cfg["value"], cfg["min_value"], cfg["max_value"], cfg["step"])
                      for cfg in state_dict["uai_params"]]
            spsa_params = SpsaParams(**state_dict["spsa_params"])
            t = state_dict["t"]
    else:
        params = params_from_config("config.json")
        spsa_params = spsa_from_config("spsa.json")

    cuteataxx = cuteataxx_from_config("cuteataxx.json")
    spsa = SpsaTuner(spsa_params, params, cuteataxx)
    spsa.t = t
    graph = Graph()

    avg_time = 0

    start_t = t

    print("Initial state: ")
    for param in spsa.params:
        print(param)
    print()
    try:
        while True:
            start = time.time()
            spsa.step()
            avg_time += time.time() - start

            graph.update(spsa.t, copy.deepcopy(spsa.params))
            graph.save("graph.png")

            if ((spsa.t / cuteataxx.games) % cuteataxx.save_rate) == 0:
                print("Saving state...")
                save_state(spsa)

            print(
                f"iterations: {int(spsa.t / cuteataxx.games)} ({((avg_time / (spsa.t - start_t)) * cuteataxx.games):.2f}s per iter)")
            print(
                f"games: {spsa.t} ({(avg_time / (spsa.t - start_t)):.2f}s per game)")
            for param in spsa.params:
                print(param)
            print()
    finally:
        print("Saving state...")
        save_state(spsa)
        print("Final results: ")
        print(
            f"iterations: {int(spsa.t / cuteataxx.games)} ({((avg_time / (spsa.t - start_t)) * cuteataxx.games):.2f}s per iter)")
        print(
            f"games: {spsa.t} ({(avg_time / (spsa.t - start_t)):.2f}s per game)")
        print("Final parameters: ")
        for param in spsa.params:
            print(param)


if __name__ == "__main__":
    main()
