import numpy as np

from .base_model import BaseModel
from src.tools.useful_funcs.step_function import SmoothStepTanh


class Viscosity(BaseModel):

    def __init__(self):
        super().__init__()
        pass

    def f(self, phi):
        raise NotImplementedError

    def df(self, phi):
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__:=^40}'


class StepViscosity(Viscosity):

    def __init__(self):
        super().__init__()
        self.func_instance = SmoothStepTanh(mode="step",
                                            step_1=self.ZETA_ECM,
                                            step_2=self.ZETA_CELL,
                                            x_mid=0.5,
                                            width=0.25)

    def f(self, phi):
        return self.func_instance.f(phi)

    def df(self, phi):
        return self.func_instance.df(phi)

    def __str__(self):
        title_line = super().__str__() + "\n"
        info = self.func_instance.__str__()
        return title_line + info


class ExpViscosity(Viscosity):
    """
    a*exp(-b*phi)
    """

    def __init__(self):
        super().__init__()
        self.a = np.exp((self.PHI_CELL * np.log(self.ZETA_ECM)
                         - self.PHI_ECM * np.log(self.ZETA_CELL))
                        / (self.PHI_CELL - self.PHI_ECM))
        self.b = np.log(self.ZETA_ECM / self.ZETA_CELL) \
                 / (self.PHI_CELL - self.PHI_ECM)

    def f(self, phi):
        return self.a * np.exp(-self.b*phi)

    def df(self, phi):
        return -self.a * self.b * np.exp(-self.b*phi)

    def __str__(self):
        title_line = super().__str__() + "\n"
        expression = "a * exp( -b * phi )\n"
        variables = f'{"":<4}a = {self.a:.4e}\n' + f'{"":<4}b = {self.b:.4e}'
        return title_line + expression + variables
