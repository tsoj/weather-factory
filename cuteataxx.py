from dataclasses import dataclass
from param import Param
import subprocess
import json
import os


@dataclass
class MatchResult:
    w: int
    l: int
    d: int
    elo_diff: float



def get_option_dictionary(params: list[Param]):
    result = {}
    for param in params:
        result[param.name] = str(param.get())
    return result

class CuteataxxMan:

    def __init__(
        self,
        engine: str,
        book: str,
        games: int = 120,
        tc: float = 5.0,
        hash: int = 8,
        threads: int = 1,
        save_rate: int = 10
    ):
        self.engine = engine
        self.book = book
        self.games = games
        self.tc = tc
        self.inc = tc / 100
        self.hash_size = hash
        self.threads = threads
        self.save_rate = save_rate

    def write_cuteataxx_settings(
        self,
        params_a: list[Param],
        params_b: list[Param]
    ) -> str:
        settings = {
            "games": self.games,
            "concurrency": self.threads,
            "verbose": False,
            "debug": False,
            "recover": False,
            "colour1": "Black",
            "colour2": "White",
            "tournament": "roundrobin",
            "print_early": False,
            "adjudicate": {
                "gamelength": 300,
                "material": 30,
                "easyfill": True,
                "timeout_buffer": 25
            },
            "openings": {
                "path": f"./tuner/{self.book}",
                "repeat": True,
                "shuffle": True
            },
            "timecontrol": {
                "time": int(self.tc*1000),
                "inc": int(self.inc*1000)
            },
            "options": {
                "debug": "False",
                "threads": "1",
                "hash": f"\"{self.hash_size}\"",
                "ownbook": "False"
            },
            "pgn": {
                "enabled": False,
                "verbose": False,
                "override": False,
                "path": "./tuner/games.pgn",
                "event": "weather-factor-ataxx"
            },
            "engines": [
                {
                    "name": f"{self.engine}_A",
                    "path": f"{os.getcwd()}/tuner/{self.engine}",
                    "protocol": "UAI",
                    "options": get_option_dictionary(params_a)
                },
                {
                    "name": f"{self.engine}_B",
                    "path": f"{os.getcwd()}/tuner/{self.engine}",
                    "protocol": "UAI",
                    "options": get_option_dictionary(params_b)
                }
            ]
        }
        filename = "./tuner/settings.json"
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
        return filename
        


    def get_cuteataxx_cmd(
        self,
        params_a: list[Param],
        params_b: list[Param]
    ) -> str:
        settings_file = self.write_cuteataxx_settings(params_a, params_b)
        return f"./tuner/cuteataxx-cli {settings_file}"

    def run(self, params_a: list[str], params_b: list[str]) -> MatchResult:
        cmd = self.get_cuteataxx_cmd(params_a, params_b)
        print(cmd)
        cuteataxx_process = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


        score = [0, 0, 0]
        elo_diff = 0.0

        for line in cuteataxx_process.stdout.splitlines():

            line = line.strip()
            print(line)
            
            # Parse WLD score
            if line.startswith("Score of"):
                start_index = line.find(":") + 1
                end_index = line.find("[")
                split = line[start_index:end_index].split(" - ")

                score = [int(i) for i in split]

            # Parse Elo Difference
            if line.startswith("Elo difference"):
                start_index = line.find(":") + 1
                end_index = line.find("+")
                elo_diff = float(line[start_index:end_index])

        print(f"Elo diff: {elo_diff}, score: {score}")
        
        return MatchResult(*score, elo_diff)
