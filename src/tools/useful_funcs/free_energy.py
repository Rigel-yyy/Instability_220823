import numpy as np
from ..type_tools import ScalarField2D
from scipy.optimize import least_squares


class LGFreeEnergy:
    """
    f(phi) = (phi-phi_0)**4 + a * (phi-phi_0)**2
    """

    def __init__(self, mode: str = "phi", **kwargs):
        self.a = None
        self.phi_0 = None
        self.phi_1 = None
        self.phi_2 = None
        if mode == "phi":
            self._phi_construct(**kwargs)
        elif mode == "param":
            self._param_construction(**kwargs)

    def _phi_construct(self, **kwargs):
        """
        mode == "phi" construct free energy from minimum
        """
        self.phi_1 = kwargs["phi_1"]
        self.phi_2 = kwargs["phi_2"]
        self.a = -(kwargs["phi_1"] - kwargs["phi_2"]) ** 2 / 2
        self.phi_0 = (kwargs["phi_1"] + kwargs["phi_2"]) / 2

    def _param_construction(self, **kwargs):
        """
        mode == "param" construct free energy from given parameter
        """
        self.a = kwargs["a"]
        self.phi_0 = kwargs["phi_0"]
        with np.errstate(invalid='ignore'):
            self.phi_1 = -np.sqrt(-self.a / 2) + self.phi_0
            self.phi_2 = np.sqrt(-self.a / 2) + self.phi_0

    def f(self, phi: ScalarField2D) -> ScalarField2D:
        """
        自由能值
        """
        return (phi - self.phi_0) ** 4 + self.a * (phi - self.phi_0) ** 2

    def df(self, phi: ScalarField2D) -> ScalarField2D:
        """
        自由能一阶导数
        """
        return 4 * (phi - self.phi_0) ** 3 + 2 * self.a * (phi - self.phi_0)

    def d2f(self, phi: ScalarField2D) -> ScalarField2D:
        """
        自由能二阶导数
        """
        return 12 * (phi - self.phi_0) ** 2 + 2 * self.a

    def __str__(self):
        expression = "( x - x0 )**4 + a * ( x - x0 )**2\n"
        variables = f'{"":<4}a = {self.a:.4e}\n' \
                    + f'{"":<4}x0 = {self.phi_0:.4e}'
        return expression + variables


class FHFreeEnergy:
    """
    f(phi) = phi*ln(phi) + (1-phi)*ln(1-phi) + chi * phi*(1-phi)
    """

    def __init__(self, mode: str = "phi", **kwargs):
        self.phi_1 = None
        self.phi_2 = None
        self.chi = None

        if mode == "phi":
            self._phi_construction(**kwargs)
        elif mode == "param":
            self._param_construction(**kwargs)

    def _phi_construction(self, **kwargs):
        self.phi_1 = kwargs["phi_1"]
        self.phi_2 = kwargs["phi_2"]
        self.chi = - np.log(self.phi_1/self.phi_2) / (self.phi_2 - self.phi_1)

    def _param_construction(self, **kwargs):
        self.chi = kwargs["chi"]
        if self.chi <= 2:
            self.phi_1 = np.nan
            self.phi_2 = np.nan
        else:
            def eqn(x):
                return self.chi * (1 - 2 * x) + np.log(x / (1 - x))

            solver = least_squares(eqn, 0.1, bounds=(0, 0.5))
            self.phi_1 = solver.x
            self.phi_2 = 1 - solver.x

    def f(self, phi: ScalarField2D) -> ScalarField2D:
        return phi * np.log(phi) + (1 - phi) * np.log(1 - phi) + self.chi * phi * (1 - phi)

    def df(self, phi: ScalarField2D) -> ScalarField2D:
        return self.chi * (1 - 2 * phi) + np.log(phi / (1 - phi))

    def d2f(self, phi: ScalarField2D) -> ScalarField2D:
        return -2 * self.chi + 1 / phi + 1 / (1 - phi)

    def __str__(self):
        expression = "x ln x + ( 1 - x ) ln( 1 - x ) + chi * x ( 1 - x )\n"
        variables = f'{"":<4}chi = {self.chi:.4e}'
        return expression + variables
