import numpy as np

from ..type_tools import ScalarField2D
from .step_function import SmoothStepTanh


class ExpDecay:
    """
    exp( -x / x0 )
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def f(self, mat: ScalarField2D):
        return np.exp(-mat / self.threshold)

    def df(self, mat: ScalarField2D):
        return -self.f(mat) / self.threshold

    def __str__(self):
        expression = "exp( -p / p0 )\n"
        variables = f'{"":<4}p0 = {self.threshold:.4e}'
        return expression + variables


class StepConstrain:
    """
    if p > pc => 0, else => 1
    """

    def __init__(self, threshold, theta_width=None):
        if theta_width is None:
            theta_width = 0.1 * threshold
        self.theta_func = SmoothStepTanh(
            mode="step",
            step_1=1,
            step_2=0,
            x_mid=threshold,
            width=theta_width
        )

    def f(self, mat: ScalarField2D):
        return self.theta_func.f(mat)

    def df(self, mat: ScalarField2D):
        return self.theta_func.df(mat)

    def __str__(self):
        return self.theta_func.__str__()


class HillConstrain:
    """
    if x < x0 => 1 - (x / x0)**H, else => 0
    """

    def __init__(self, threshold, poly_index=4):
        self.threshold = threshold
        self.poly_index = poly_index

    def _hill(self, mat: ScalarField2D):
        return 1 - (mat / self.threshold)**self.poly_index

    def _dhill(self, mat: ScalarField2D):
        return -self.poly_index * (mat / self.threshold)**(self.poly_index - 1) / self.threshold

    def f(self, mat: ScalarField2D):
        result = self._hill(mat)
        return np.where(mat < self.threshold, result, 0)

    def df(self, mat: ScalarField2D):
        result = self._dhill(mat)
        return np.where(mat < self.threshold, result, 0)

    def __str__(self):
        expression = "if p < p0 => 1 - ( p / p0 )**H; else => 0\n"
        variables = f'{"":<4}H = {self.poly_index:.4e}\n' + \
                    f'{"":<4}p0 = {self.threshold:.4e}'
        return expression + variables
