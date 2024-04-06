# weather-factory

Weather Factory is a WIP SPSA tuner for UAI compliant Ataxx engines, forked from jnlt3's [Weather Factory](https://github.com/jnlt3/weather-factory) for chess engines.

Usage instructions:

make a folder "tuner"
Put the cuteataxx-cli binary, the opening book and the engine binary in the folder.

Change the .json config files following the format presented.

Each parameter in config.json should correspond to one UAI parameter:

```
option name TEST type spin default 100 min 50 max 150
```
```json
"TEST": {
    "value": 100,
    "min_value": 50,
    "max_value": 150,
    "step": 10
}
```
the step parameter should be large enough to create a 2-3 elo difference.

For the SPSA config file, none of the values used in spsa.json except "A" require changing. Ideally `A = max iterations / 10`.

cuteataxx.json:
```
engine: name of the engine inside the `tuner` folder

book: name of the book inside the `tuner` folder

games: number of games played per SPSA iteration, make sure this is a multiple of two and ideally a multiple of threads.

tc: The time control the matches are going to be played at. Increment is automatically `tc / 100`.

threads: This corresponds to concurrency it cuteataxx, not the threads of the engine.

save_rate: The number of games between times saving the state to a file
```

