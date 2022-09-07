# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:07:26 2022

@author: Rigel_yyy
"""

import pickle
from typing import TYPE_CHECKING
from timeit import default_timer as timer
from rich.progress import Progress
import numpy as np

from src.tools.num_tools import inner2D, has_nan
from src.tools import data_tools
from src.runtime_sim_log import RuntimeLogging
from .base_model import BaseModel
from .pressure_solver import IterativePressureSolver

if TYPE_CHECKING:
    from src.tools.pde_tools import OperatorLib
    from .active_growth import ActiveGrowth
    from .initial_condition import InitialCondition
    from .pressure_solver import PressureSolver
    from .viscosity import Viscosity
    from .view_move import ViewMove


class ModelLogging:
    setting_file = "settings.txt"

    @classmethod
    def _log_title(cls, title_name):
        print(f'{title_name:=^40}', file=open(cls.setting_file, "a+"))

    @classmethod
    def log_boundary_condition(cls, bc_name: str, bc: "OperatorLib"):
        cls._log_title(bc_name)

        print("boundary_type:", file=open(cls.setting_file, "a+"))
        for key, value in bc.bound_condition.items():
            print(f'{"":<4}{key:<20}:{value}', file=open(cls.setting_file, "a+"))

        print("boundary_value:", file=open(cls.setting_file, "a+"))
        for key, value in bc.bound_value.items():
            print(f'{"":<4}{key:<20}:{value:.4e}', file=open(cls.setting_file, "a+"))

    @classmethod
    def log_sim_function(cls, obj, title_name=None):
        # obj必须有__str__方法
        if title_name:
            cls._log_title(title_name)
        print(obj, file=open(cls.setting_file, "a+"))


class ModelStatus(BaseModel):
    """
    存储并管理模拟中的状态变量
    """

    def __init__(self):
        super().__init__()
        self.field_keys = ["phi", "grad_phi",
                           "pressure", "grad_p",
                           "v_cell", "v_avg",
                           "zeta",
                           "mu", "osmotic"]
        self.phi = self.create_empty_scalar()
        self.grad_phi = self.create_empty_vector()
        self.pressure = self.create_empty_scalar()
        self.grad_p = self.create_empty_vector()
        self.v_cell = self.create_empty_vector()
        self.v_avg = self.create_empty_vector()
        self.zeta = self.create_empty_scalar()
        self.mu = self.create_empty_scalar()
        self.osmotic = self.create_empty_vector()

        if self.GROWTH_RATE != 0:
            self.growth = self.create_empty_scalar()
            self.field_keys.append("growth")

        self.t_init = 0
        self.view_init = 0

        self.frame_now = 0
        self.sim_time = 0
        self.view_moved = 0

        self.sim_time_frame = []
        self.view_moved_frame = []

    def create_empty_scalar(self):
        return np.zeros((self.N_ROW, self.N_COLUMN))

    def create_empty_vector(self):
        return np.zeros((2, self.N_ROW, self.N_COLUMN))

    def save_current_frame(self):
        for name in self.field_keys:
            field = getattr(self, name)
            data_tools.npz_append_save(name, str(self.frame_now), field)

        self.sim_time_frame.append(self.sim_time + self.t_init)
        self.view_moved_frame.append(self.view_moved + self.view_init)
        self.frame_now += 1

    def save_frame_info(self):
        file_name = "frame_info.npz"
        np.savez(file_name,
                 sim_time=np.array(self.sim_time_frame),
                 view_moved=np.array(self.view_moved_frame))

    def save_config_info(self):
        config_data = self.get_config()
        data_tools.save_config(config_data)

    def load_frame(self, n: int):
        for name in self.field_keys:
            with np.load(name + ".npz") as data:
                field = data[str(n)]
                setattr(self, name, field)

        if "growth" not in self.field_keys:
            self.growth = self.create_empty_scalar()

        self.sim_time = self.sim_time_frame[n]
        self.view_moved = self.view_moved_frame[n]
        self.frame_now = n

    def load_frame_info(self):
        with np.load("frame_info.npz") as data:
            self.sim_time_frame = list(data["sim_time"])
            self.view_moved_frame = list(data["view_moved"])

    def copy_status(self, src_model: "ModelStatus", cont=True):
        for name in self.field_keys:
            field = getattr(src_model, name)
            setattr(self, name, field)

        if cont:
            self.t_init = src_model.sim_time
            self.view_init = src_model.view_moved


class TwoFluidModel(ModelStatus):
    """
    on lattice grid field construction
    """

    def __init__(self, builder):
        super().__init__()
        self.initial_condition: InitialCondition = builder.initial_condition
        self.phi_bound: "OperatorLib" = builder.phi_bound
        self.mu_bound: "OperatorLib" = builder.mu_bound
        self.p_bound: "OperatorLib" = builder.p_bound
        self.free_nrg = builder.free_nrg
        self.viscosity: "Viscosity" = builder.viscosity
        self.growth_func: "ActiveGrowth" = builder.growth_func
        self.p_solver: "PressureSolver" = builder.p_solver
        self.view_move_engine: "ViewMove" = builder.view_move_engine
        self.sim_termination: bool = builder.sim_termination
        self.logging = ModelLogging()

    def save_model(self):
        with open('model.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

        self.logging.log_boundary_condition("phi_boundary", self.phi_bound)
        self.logging.log_boundary_condition("mu_boundary", self.mu_bound)
        self.logging.log_boundary_condition("pressure_boundary", self.p_bound)
        self.logging.log_sim_function(self.free_nrg,
                                      title_name=self.free_nrg.__class__.__name__)
        self.logging.log_sim_function(self.viscosity)
        self.logging.log_sim_function(self.growth_func)
        self.logging.log_sim_function(self.p_solver)
        self.logging.log_sim_function(self.view_move_engine)

    def set_init(self):
        self.phi = self.initial_condition.set_phi_init()

    def solvePDE(self):
        frame_step = self.T_FINAL / (self.N_FRAME - 1)
        self.save_config_info()

        with Progress() as progress:

            RuntimeLogging().info("Start solving PDE")
            start = timer()

            task = progress.add_task("[cyan]Solving PDE", total=self.T_FINAL)
            try:
                total_n_step = int(self.T_FINAL / self.T_STEP) + 1
                for step in range(total_n_step):
                    self.sim_time = step * self.T_STEP

                    self.update_view_move()
                    self.update_zeta()
                    self.update_mu()
                    self.update_osmotic()
                    self.nan_check()
                    self.update_growth()
                    self.update_grad_phi()
                    self.update_pressure()
                    self.update_grad_p()
                    self.update_velocity()
                    self.update_phi()

                    if self.sim_time >= frame_step * self.frame_now:
                        self.save_current_frame()
                        progress.update(task, completed=self.sim_time)
                    if self.loop_terminate():
                        msg = f'Boundary reached at t = {self.sim_time:.5e}, loop termination triggered.'
                        RuntimeLogging().warning(msg)
                        break

                self.sim_time += self.T_STEP
                self.save_current_frame()
                progress.update(task, completed=min(self.T_FINAL, self.sim_time))

                self.save_frame_info()

            except (FloatingPointError, RuntimeError):
                self.save_current_frame()
                self.save_frame_info()
                msg = f'Simulation ended at t = {self.sim_time:.5e} with {self.frame_now} frames saved.'
                RuntimeLogging().error(msg, exc_info=True)
                raise

            end = timer()
            ms_per_iter = 1000 * (end - start) / total_n_step
            RuntimeLogging().info(f"PDE solved successfully. {ms_per_iter:.4f} ms per time step.")

    def update_view_move(self):
        self.view_move_engine.set_move_direction(self.phi)
        self.phi = self.view_move_engine.phi_view_move(self.phi)
        self.pressure = self.view_move_engine.p_view_move(self.pressure)
        self.view_moved += self.view_move_engine.move_direction

    def update_zeta(self):
        self.zeta = self.viscosity.f(self.phi)

    def update_mu(self):
        del2_phi = self.phi_bound.laplace(self.phi)
        self.mu = self.NRG * self.free_nrg.df(self.phi) - self.SURF * del2_phi

    def update_osmotic(self):
        # phi grad(mu)
        self.osmotic = self.phi * self.mu_bound.grad(self.mu)

    def nan_check(self):
        nan_idx = has_nan(self.osmotic.sum(axis=0))

        if nan_idx is not None:
            x_idx = nan_idx % self.N_COLUMN
            y_idx = nan_idx // self.N_COLUMN
            msg = f'Nan occurred at (x_idx = {x_idx}, y_idx = {y_idx}) when calculating osmotic force.'
            msg += f'Phi[{y_idx},{x_idx}] = {self.phi[y_idx, x_idx]:.4f}.'
            RuntimeLogging().error(msg)

            raise FloatingPointError

    def update_growth(self):
        # 对于生长受到压强限制的情形，需要在算出压强后再更新一次self.growth
        self.growth = self.growth_func.get_growth_term(phi=self.phi, p=self.pressure)

    def update_grad_phi(self):
        # grad(phi)
        self.grad_phi = self.phi_bound.grad(self.phi)

    def update_pressure(self):
        """
        解方程：del2(p) + h*grad(p) = g0 + g1 * lambda(p)
        lambda(p)表示生长项
        """

        # h = - grad(zeta) / zeta = - (d_zeta/zeta) * grad(phi)
        h = -self.viscosity.df(self.phi) / self.zeta
        h = self.grad_phi * h

        # div(f) = grad(phi)*grad(mu) + phi*del2(mu)
        div_f = inner2D(self.phi_bound.grad(self.phi),
                        self.mu_bound.grad(self.mu))
        div_f += self.phi * self.mu_bound.laplace(self.mu)

        # g = -lambda(phi)*zeta - div(f) + grad(zeta)*f / zeta
        g0 = -div_f - inner2D(h, self.osmotic)
        g1 = -self.zeta

        self.pressure = self.p_solver.solve(h, g0, g1, p0=self.pressure)

        if isinstance(self.p_solver, IterativePressureSolver):
            self.update_growth()

    def update_grad_p(self):
        # grad(p)
        self.grad_p = self.p_bound.grad(self.pressure)

    def update_velocity(self):
        """
        v = -(grad p + f) / zeta
        """
        self.v_avg = -(self.grad_p + self.osmotic) / self.zeta
        self.v_cell = self.v_avg - (1 - self.phi) ** 2 * self.osmotic / self.XI

    def update_phi(self):
        """
        d/dt phi + div(phi*v_cell) = lambda(phi)
        d/dt phi + v grad(phi) = (1-phi)lambda(phi) + (1-phi^2)phi^2 del2(mu)/ xi
                                 + 2phi(1-phi)(1-2phi) grad(phi) grad(mu) / xi
        """
        del2_mu = self.mu_bound.laplace(self.mu)

        d_phi = -inner2D(self.grad_phi, self.v_avg)
        d_phi += (1 - self.phi) ** 2 * self.phi ** 2 * del2_mu / self.XI
        d_phi += 2 * (1 - self.phi) * (1 - 2 * self.phi) * \
                 inner2D(self.grad_phi, self.osmotic) / self.XI

        if self.GROWTH_RATE:
            d_phi += self.growth * (1 - self.phi)

        self.phi += self.T_STEP * d_phi

    def loop_terminate(self) -> bool:
        if self.sim_termination:
            return np.count_nonzero(self.phi[:, -3:] > 0.5) >= 5
        else:
            return False
