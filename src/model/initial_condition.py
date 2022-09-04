# -*- coding: utf-8 -*-
"""
Created on Thu Aug 1 14:08:56 2022

@author: Rigel_yyy
"""

import numpy as np

from src.tools.useful_funcs.perturbation import (PerturbType,
                                                 SingleSine,
                                                 MultipleSine)
from src.tools.useful_funcs.step_function import (SmoothStepTanh,
                                                  SmoothSquareTanh)
from .base_model import BaseModel


class InitialCondition(BaseModel):

    def __init__(self, perturb_type: PerturbType,
                 noise: float,
                 period: int = None,
                 max_period: int = None):
        super().__init__()
        self.perturb_type = perturb_type
        self.transition = 2 * np.sqrt(self.SURF / self.NRG)
        self.noise_func = self.get_noise_func(noise, period, max_period)

    def get_noise_func(self, noise: float,
                       period: int = None,
                       max_period: int = None):
        if self.perturb_type == PerturbType.SINGLE_SINE:
            wave_num = 2 * np.pi * period / (self.N_ROW - 1)
            return SingleSine(noise, wave_num)
        elif self.perturb_type == PerturbType.MULTI_SINE:
            wave_num_arr = 2 * np.pi * np.arange(max_period) / (self.N_ROW - 1)
            return MultipleSine(noise, wave_num_arr)

    def set_phi_init(self):
        raise NotImplementedError


class HalfPlaneInit(InitialCondition):
    """
    initial condition with boundary at x=x0
    """

    def __init__(self, x0=None, **kwargs):
        if x0 is None:
            self.x0 = int(self.N_COLUMN / 2)
        else:
            self.x0 = x0
        super().__init__(**kwargs)

    def set_phi_init(self):
        phi_profile = SmoothStepTanh(
            mode="step",
            step_1=self.PHI_CELL,
            step_2=self.PHI_ECM,
            x_mid=0,
            width=self.transition
        )
        _, dist = np.indices((self.N_ROW, self.N_COLUMN)).astype(float)

        wave_form = self.noise_func.f(np.arange(self.N_ROW))

        dist += wave_form[:, np.newaxis]
        dist_mat = (dist - self.x0) * self.GRID
        return phi_profile.f(dist_mat)


class MiddleInit(InitialCondition):
    """
    initial condition with cell in the mid
    x_mid: center_x of the tissue
    width: 1/2 total width of the tissue
    """

    def __init__(self, width, x_mid=None, **kwargs):
        self.width = width
        if x_mid is None:
            self.x_mid = self.N_COLUMN / 2
        else:
            self.x_mid = x_mid

        super().__init__(**kwargs)

    def set_phi_init(self):
        phi_profile = SmoothSquareTanh(
            mode="step",
            step_1=self.PHI_ECM,
            step_2=self.PHI_CELL,
            l_bound=(self.x_mid - self.width) * self.GRID,
            r_bound=(self.x_mid + self.width) * self.GRID,
            width=self.transition
        )
        _, dist = np.indices((self.N_ROW, self.N_COLUMN)).astype(float)

        l_wave_form = self.noise_func.f(np.arange(self.N_ROW), reset=True)
        r_wave_form = self.noise_func.f(np.arange(self.N_ROW), reset=True)

        dist[:, :self.x_mid] += l_wave_form[:, np.newaxis]
        dist[:, self.x_mid:] += r_wave_form[:, np.newaxis]
        dist_mat = dist * self.GRID
        return phi_profile.f(dist_mat)
