# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:15:37 2022

@author: Rigel_yyy
"""

import numpy as np

from src.tools.type_tools import ScalarField2D
from src.tools.useful_funcs.step_function import SmoothSquareTanh
from src.tools.useful_funcs.rate_constrain import (ExpDecay,
                                                   StepConstrain,
                                                   HillConstrain)
from .base_model import BaseModel


class ActiveGrowth(BaseModel):

    def __init__(self, phi_threshold=0.65):
        super().__init__()
        self.growth_func = SmoothSquareTanh(
            mode="step",
            step_1=0,
            step_2=self.GROWTH_RATE,
            l_bound=self.PHI_CELL - 0.005,
            r_bound=self.PHI_CELL + phi_threshold,
            width=0.005
        )
        self.recent_f = None  # 保存最近一次计算得到的growth rate

    def get_growth_term(self, phi: ScalarField2D, **kwargs):
        raise NotImplementedError

    def get_recent_f(self):
        return self.recent_f

    def __str__(self):
        title_line = f'{self.__class__.__name__:=^40}' + "\n"
        growth_title = "[Growth Function] \n"
        info = self.growth_func.__str__()
        return title_line + growth_title + info


class OffGrowth(ActiveGrowth):

    def get_growth_term(self, phi: ScalarField2D, **kwargs):
        self.recent_f = 0
        return 0

    def __str__(self):
        return f'{self.__class__.__name__:=^40}'


class FullGrowth(ActiveGrowth):

    def get_growth_term(self, phi: ScalarField2D, **kwargs):
        self.recent_f = self.growth_func.f(phi)
        return self.recent_f


class InitConstrainedGrowth(ActiveGrowth):

    def __init__(self, rate_threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.growth_mask = None
        self.rate_threshold = self.set_threshold(rate_threshold)
        self.recent_full_f = None  # 保存最近一次计算所得的完全生长状态的值

    def set_threshold(self, rate_threshold):
        if rate_threshold is None:
            return 0.05 * self.GROWTH_RATE
        else:
            return rate_threshold

    def get_growth_term(self, phi: ScalarField2D, **kwargs):
        self.recent_full_f = self.growth_func.f(phi)

        if self.growth_mask is None:
            self.growth_mask = self.recent_full_f > self.rate_threshold

        self.recent_f = np.where(self.growth_mask, self.recent_full_f, 0)
        return self.recent_f

    def __str__(self):
        growth_info = super().__str__() + "\n"
        constrain_info = f'rate_threshold phi for initial growth: {self.rate_threshold:.4e}'
        return growth_info + constrain_info


class PressureConstrainedGrowth(ActiveGrowth):

    def __init__(self, form: str,
                 p_threshold: float,
                 theta_width_ratio: float = None,
                 H: int = None, **kwargs):
        super().__init__(**kwargs)
        self.constrain_func = self.get_p_constrain(form, p_threshold, theta_width_ratio, H)
        self.recent_full_f = None  # 保存最近一次计算所得的完全生长状态的值

    def calc_full_growth(self, phi: ScalarField2D):
        self.recent_full_f = self.growth_func.f(phi)
        return self.recent_full_f

    def get_growth_term(self, phi: ScalarField2D, **kwargs):
        p_mat = kwargs["p"]
        self.calc_full_growth(phi)
        self.recent_f = self.constrain_func.f(p_mat) * self.recent_full_f
        return self.recent_f

    @staticmethod
    def get_p_constrain(form: str, p_threshold: float, theta_width_ratio: float, H: int):
        if form == "exp":
            return ExpDecay(threshold=p_threshold)

        elif form == "step":
            if theta_width_ratio is not None:
                return StepConstrain(threshold=p_threshold,
                                     theta_width_ratio=theta_width_ratio)
            else:
                return StepConstrain(threshold=p_threshold)

        elif form == "poly":
            if H is not None:
                return HillConstrain(threshold=p_threshold,
                                     poly_index=H)
            else:
                return HillConstrain(threshold=p_threshold)

    # 设计有待改进
    def d_dp_growth(self, p: ScalarField2D, phi: ScalarField2D = None):
        if phi is None:
            return self.recent_full_f * self.constrain_func.df(p)
        else:
            return self.growth_func.f(phi) * self.constrain_func.df(p)

    def __str__(self):
        growth_info = super().__str__() + "\n"
        constrain_title = "[Pressure Constrain Function] \n"
        constrain_info = self.constrain_func.__str__()
        return growth_info + constrain_title + constrain_info
