from cuteataxx import CuteataxxMan
from dataclasses import dataclass
from random import randint
import copy
from param import Param

@dataclass
class SpsaParams:
    a: float
    c: float
    A: int
    alpha: float = 0.601
    gamma: float = 0.102


class SpsaTuner:

    def __init__(
        self,
        spsa_params: SpsaParams,
        uai_params: list[Param],
        cuteataxx: CuteataxxMan
    ):
        self.uai_params = uai_params
        self.spsa = spsa_params
        self.cuteataxx = cuteataxx
        self.delta = [0] * len(uai_params)
        self.t = 0

    def step(self):
        self.t += self.cuteataxx.games
        a_t = self.spsa.a / (self.t + self.spsa.A) ** self.spsa.alpha
        c_t = self.spsa.c / self.t ** self.spsa.gamma

        self.delta = [randint(0, 1) * 2 - 1 for _ in range(len(self.delta))]

        uai_params_a = []
        uai_params_b = []
        for param, delta in zip(self.uai_params, self.delta):
            curr_delta = param.step

            step = delta * curr_delta * c_t

            uai_a = copy.deepcopy(param)
            uai_b = copy.deepcopy(param)

            uai_a.update(step)
            uai_b.update(-step)

            uai_params_a.append(uai_a)
            uai_params_b.append(uai_b)

        gradient = self.gradient(uai_params_a, uai_params_b)

        for param, delta, param in zip(self.uai_params, self.delta, self.uai_params):
            param_grad = gradient / (delta * c_t)
            param.update(-param_grad * a_t * param.step)

    @property
    def params(self) -> list[Param]:
        return self.uai_params

    def gradient(self, params_a: list[Param], params_b: list[Param]) -> float:
        game_result = self.cuteataxx.run(params_a, params_b)
        return (game_result.l - game_result.w)
