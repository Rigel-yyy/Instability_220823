# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:51:09 2022

@author: Rigel_yyy
"""

from .base_model import BaseModel
from .initial_condition import MiddleInit, HalfPlaneInit
from src.tools.useful_funcs.free_energy import (LGFreeEnergy,
                                                FHFreeEnergy)
from .viscosity import StepViscosity, ExpViscosity
from .active_growth import (FullGrowth,
                            OffGrowth,
                            InitConstrainedGrowth,
                            PressureConstrainedGrowth)
from .pressure_solver import (SimplifiedPressureSolver,
                              TraditionalPressureSolver,
                              IterativePressureSolver)
from .view_move import (OffViewMove,
                        HoldTissueCenter)
from .two_fluid_model import TwoFluidModel
from src.tools.pde_tools import OperatorLib, Linear2ndPDESolver


class ModelBuilder(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.initial_condition = None
        self.phi_bound = None
        self.mu_bound = None
        self.p_bound = None
        self.free_nrg = None
        self.viscosity = None
        self.growth_func = None
        self.p_solver = None
        self.view_move_engine = None

    def set_phi_bound_condition(self, bound_type, bound_value):
        shape = (self.N_ROW, self.N_COLUMN)
        self.phi_bound = OperatorLib(
            bound_value=bound_value,
            shape=shape,
            bound_type=bound_type,
            grid_size=self.GRID)

    def set_mu_bound_condition(self, bound_type, bound_value):
        shape = (self.N_ROW, self.N_COLUMN)
        self.mu_bound = OperatorLib(
            bound_value=bound_value,
            shape=shape,
            bound_type=bound_type,
            grid_size=self.GRID)

    def set_p_bound_condition(self, bound_type, bound_value):
        shape = (self.N_ROW, self.N_COLUMN)
        self.p_bound = Linear2ndPDESolver(
            stencil=5,
            bound_value=bound_value,
            shape=shape,
            bound_type=bound_type,
            grid_size=self.GRID)

    def set_initial_condition(self, init_shape: str, **kwargs):
        """
        e.g. builder.set_initial_condition(init_shape = "middle",
                                           x_mid = 40, width = 20,
                                           perturb_type = PerturbType.WHITE_NOISE,
                                           noise = 1, max_period = 8)
        """

        if init_shape == "middle":
            self.initial_condition = MiddleInit(**kwargs)
        elif init_shape == "half_plane":
            self.initial_condition = HalfPlaneInit(**kwargs)

    def set_free_energy(self, form: str):
        if form == "FH":
            self.free_nrg = FHFreeEnergy(mode="phi",
                                         phi_1=self.PHI_ECM,
                                         phi_2=self.PHI_CELL)
        elif form == "LG":
            self.free_nrg = LGFreeEnergy(mode="phi",
                                         phi_1=self.PHI_ECM,
                                         phi_2=self.PHI_CELL)

    def set_viscosity(self, form: str = "step"):
        if form == "step":
            self.viscosity = StepViscosity()

        elif form == "exp":
            self.viscosity = ExpViscosity()

    def set_growth_function(self, constrain_type: str = None, **kwargs):
        """
        e.g. builder.set_growth_function(constrain_type = None)
             builder.set_growth_function(constrain_type = "init", threshold = None)
             builder.set_growth_function(constrain_type = "pressure",
                                         form = "step",
                                         threshold = 10)
        """
        if not self.GROWTH_RATE:
            self.growth_func = OffGrowth()
        else:
            if constrain_type is None:
                self.growth_func = FullGrowth()
            elif constrain_type == "init":
                self.growth_func = InitConstrainedGrowth(**kwargs)
            elif constrain_type == "pressure":
                self.growth_func = PressureConstrainedGrowth(**kwargs)

    def set_pressure_solver(self, solve_method: str, **kwargs):
        """
        e.g. builder.set_pressure_solver(solve_method = "iterative")
        """

        # 要求已经设置过压强的边界条件了
        if self.p_bound is None:
            raise ValueError("Pressure boundary condition undefined!")
        # 要求已经设置过生长项
        if self.growth_func is None:
            raise ValueError("Growth term undefined!")

        if isinstance(self.growth_func, PressureConstrainedGrowth):
            if solve_method != "iterative":
                raise ValueError("Iterative pressure solver required!")

            self.p_solver = IterativePressureSolver(solver=self.p_bound,
                                                    growth_func=self.growth_func,
                                                    **kwargs)
            return

        if solve_method == "simplified":
            self.p_solver = SimplifiedPressureSolver(solver=self.p_bound,
                                                     growth_func=self.growth_func)
        elif solve_method == "traditional":
            self.p_solver = TraditionalPressureSolver(solver=self.p_bound,
                                                      growth_func=self.growth_func)

    def set_view_move(self, engine="off"):
        if self.phi_bound is None:
            raise ValueError("Phi boundary condition undefined!")
        elif self.p_bound is None:
            raise ValueError("Pressure boundary condition undefined!")

        if engine == "off":
            self.view_move_engine = OffViewMove(self.phi_bound, self.p_bound)
        elif engine == "tissue_center":
            self.view_move_engine = HoldTissueCenter(self.phi_bound, self.p_bound)

    def build_model(self):
        return TwoFluidModel(self)
