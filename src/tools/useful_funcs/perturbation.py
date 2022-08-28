from enum import Enum
import numpy as np


class PerturbType(Enum):
    SINGLE_SINE = 0
    MULTI_SINE = 1


class SingleSine:
    """
    return A sin( k * s + phi) with a random phi
    """
    def __init__(self, noise, wave_num):
        self.noise = noise
        self.wave_num = wave_num
        self.phi = None
        self.set_phase()

    def set_phase(self):
        self.phi = 2 * np.pi * np.random.rand(1)

    def f(self, mat, reset=False):
        result = self.noise * np.sin(mat * self.wave_num + self.phi)
        if reset:
            self.set_phase()
        return result


class MultipleSine:
    """
    return A Sum sin( k_i * s + phi_i) with a random phi
    """
    def __init__(self, noise, wave_num_arr):
        self.noise = noise
        self.wave_num_arr = wave_num_arr
        self.n_wave_nums = len(wave_num_arr)
        self.phi_arr = None
        self.set_phase()

    def set_phase(self):
        self.phi_arr = 2 * np.pi * np.random.rand(self.n_wave_nums)

    def f(self, mat, reset=False):
        result = 0
        for wave_num, phi in zip(self.wave_num_arr, self.phi_arr):
            result += self.noise * np.sin(mat * wave_num + phi) / self.n_wave_nums
        if reset:
            self.set_phase()
        return result
