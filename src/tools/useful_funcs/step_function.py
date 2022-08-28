import numpy as np
from ..type_tools import ScalarField2D


class SmoothStepTanh:
    """
    smooth stepwise function using tanh
    a * tanh(k*(x-x0)) + b
    """
    def __init__(self, mode: str = "step", **kwargs):
        self.a = None
        self.k = None
        self.x0 = None
        self.b = None
        if mode == "step":
            self._step_construction(**kwargs)
        elif mode == "param":
            self._param_construction(**kwargs)

    def _step_construction(self, **kwargs):
        """
        mode == 'step'
        construct from step properties: step_1, bond_1, step_2, bond_2
        """
        self.a = (kwargs["step_2"] - kwargs["step_1"]) / 2
        self.b = (kwargs["step_2"] + kwargs["step_1"]) / 2
        self.x0 = kwargs["x_mid"]
        self.k = 2 / (kwargs["width"])

    def _param_construction(self, **kwargs):
        """
        mode == 'param'
        construct from a, k, x0, b
        """
        self.a = kwargs["a"]
        self.k = kwargs["k"]
        self.x0 = kwargs["x0"]
        self.b = kwargs["b"]

    def f(self, mat: ScalarField2D) -> ScalarField2D:
        return self.a * np.tanh(self.k * (mat - self.x0)) + self.b

    def df(self, mat: ScalarField2D) -> ScalarField2D:
        return 2 * self.a * self.k / (1 + np.cosh(2 * self.k * (mat - self.x0)))

    def __str__(self):
        expression = "a * tanh( k * (x - x0) ) + b\n"
        variables = f'{"":<4}a = {self.a:.4e}\n' \
                    + f'{"":<4}k = {self.k:.4e}\n' \
                    + f'{"":<4}x0 = {self.x0:.4e}\n' \
                    + f'{"":<4}b = {self.b:.4e}'
        return expression + variables


class SmoothStepSigma:
    """
    smooth stepwise function using sigma function:
    a / [1 + exp( -k * (x-x0) )] + b
    """

    def __init__(self, mode: str = "step", **kwargs):
        self.a = None
        self.k = None
        self.x0 = None
        self.b = None
        if mode == "step":
            self._step_construction(**kwargs)
        elif mode == "param":
            self._param_construction(**kwargs)

    def _step_construction(self, **kwargs):
        """
        mode == 'step'
        construct from step properties: step_1, bond_1, step_2, bond_2
        """
        self.a = kwargs["step_2"] - kwargs["step_1"]
        self.b = kwargs["step_1"]
        self.x0 = kwargs["x_mid"]
        self.k = 1 / kwargs["width"]

    def _param_construction(self, **kwargs):
        """
        mode == 'param'
        construct from a, k, x0, b
        """
        self.a = kwargs["a"]
        self.k = kwargs["k"]
        self.x0 = kwargs["x0"]
        self.b = kwargs["b"]

    def f(self, mat: ScalarField2D):
        return self.b + self.a / (np.exp(-self.k * (mat - self.x0)) + 1)

    def df(self, mat: ScalarField2D):
        temp = np.exp(-self.k * (mat - self.x0))
        return self.a * self.k * temp / (1 + temp)**2

    def __str__(self):
        expression = "a / [1 + exp( -k * (x-x0) )] + b\n"
        variables = f'{"":<4}a = {self.a:.4e}\n' \
                    + f'{"":<4}k = {self.k:.4e}\n' \
                    + f'{"":<4}x0 = {self.x0:.4e}\n' \
                    + f'{"":<4}b = {self.b:.4e}'
        return expression + variables


class SmoothSquareTanh:
    """
    smooth square wave function using tanh:
    a * tanh(k*abs(x-x0) + d) + b
    """

    def __init__(self, mode: str = "step", **kwargs):
        self.a = None
        self.b = None
        self.x0 = None
        self.k = None
        self.d = None
        if mode == "step":
            self._step_construction(**kwargs)
        elif mode == "param":
            self._param_construction(**kwargs)

    def _step_construction(self, **kwargs):
        """
        mode == 'step' construct from step properties:
        step_1, step_2, l_bond, r_bond, width
        """
        self.a = (kwargs["step_2"] - kwargs["step_1"]) / 2
        self.b = (kwargs["step_2"] + kwargs["step_1"]) / 2
        self.x0 = (kwargs["l_bound"] + kwargs["r_bound"]) / 2
        self.k = -2 / kwargs["width"]
        self.d = (kwargs["r_bound"] - kwargs["l_bound"]) / kwargs["width"]

    def _param_construction(self, **kwargs):
        """
        mode == 'param' construct from a, k, x0, d, b
        """
        self.a = kwargs["a"]
        self.b = kwargs["b"]
        self.x0 = kwargs["x0"]
        self.k = kwargs["k"]
        self.d = kwargs["d"]

    def f(self, mat: ScalarField2D):
        return self.a * np.tanh(self.k * np.abs(mat - self.x0) + self.d) + self.b

    def df(self, mat: ScalarField2D):
        return 2 * self.a * self.k * np.sign(mat - self.x0) \
               / (1 + np.cosh(2 * self.d + 2 * self.k * (np.abs(mat - self.x0))))

    def __str__(self):
        expression = "a * tanh( k * abs(x - x0) + d ) + b\n"
        variables = f'{"":<4}a = {self.a:.4e}\n' \
                    + f'{"":<4}k = {self.k:.4e}\n' \
                    + f'{"":<4}x0 = {self.x0:.4e}\n' \
                    + f'{"":<4}d = {self.d:.4e}\n' \
                    + f'{"":<4}b = {self.b:.4e}'
        return expression + variables
