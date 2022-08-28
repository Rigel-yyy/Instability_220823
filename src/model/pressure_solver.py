# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:04:15 2022

@author: Rigel_yyy
"""
from typing import TYPE_CHECKING
import numpy as np
from scipy.sparse.linalg import spsolve

from src.tools.num_tools import inner2D
from src.tools.type_tools import ScalarField2D, VectorField2D
from src.runtime_sim_log import RuntimeLogging

from .base_model import BaseModel

if TYPE_CHECKING:
    from src.tools.pde_tools import Linear2ndPDESolver
    from .active_growth import ActiveGrowth, PressureConstrainedGrowth


class PressureSolver(BaseModel):
    """
    解方程：del2(p) + h*grad(p) = g0 + g1 * lambda(p)
    lambda(p)表示生长项
    """

    def __init__(self,
                 solver: "Linear2ndPDESolver",
                 growth_func: "ActiveGrowth"):
        super().__init__()
        self.solver = solver
        self.growth_func = growth_func
        self.inner_IJV = None
        self.bound_IJV = self.solver.lhs_sps_pde_boundary()

    def solve(self,
              h: ScalarField2D,
              g0: ScalarField2D,
              g1: ScalarField2D,
              *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__:=^40}'


class TraditionalPressureSolver(PressureSolver):
    """
    solve: del2 p + h.grad p = g
    其中g = g0 + g1 * lambda
    """

    def solve(self, h: VectorField2D,
              g0: ScalarField2D,
              g1: ScalarField2D,
              *args, **kwargs):
        sim_shape = self.solver.sim_shape
        self.inner_IJV = self.solver.lhs_sps_pde_inner(A=1, B=0, C=1,
                                                       D=h[0], E=h[1], F=0)
        lhs_sps = self.solver.assemble_lhs_sps(self.inner_IJV, self.bound_IJV)
        g = g0 + g1 * self.growth_func.get_recent_f()
        rhs = self.solver.assemble_rhs(g)

        p_sim = spsolve(lhs_sps, rhs).reshape(sim_shape)
        return self.solver.sim_result_to_sol(p_sim)


class SimplifiedPressureSolver(PressureSolver):
    """
    solve: del2 p = g - h.grad p0
    其中g = g0 + g1 * lambda
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p0_grad: VectorField2D = None
        self.lhs_sps = None

    def solve(self, h: VectorField2D,
              g0: ScalarField2D,
              g1: ScalarField2D,
              *args, **kwargs):

        sim_shape = self.solver.sim_shape
        if self.lhs_sps is None:
            p_solved = self.initial_solve(h, g0, g1)
            self.p0_grad = self.solver.grad(p_solved)
            self.inner_IJV = self.solver.lhs_sps_pde_inner(A=1, B=0, C=1, D=0, E=0, F=0)
            self.lhs_sps = self.solver.assemble_lhs_sps(self.inner_IJV, self.bound_IJV)
            return p_solved
        else:
            # 计算 g - h.grad p0
            g = g0 + g1 * self.growth_func.get_recent_f()
            rhs_mat = g - inner2D(self.p0_grad, h)
            rhs = self.solver.assemble_rhs(rhs_mat)

            # 求解p1, 并更新至p0_grad
            p_sim = spsolve(self.lhs_sps, rhs).reshape(sim_shape)
            p_solved = self.solver.sim_result_to_sol(p_sim)
            self.p0_grad = self.solver.grad(p_solved)

            return p_solved

    def initial_solve(self, h: VectorField2D,
                      g0: ScalarField2D,
                      g1: ScalarField2D):
        """
        在初始时刻求解: del2 p + h.grad p = g
        g = g0 + g1 * lambda
        """

        sim_shape = self.solver.sim_shape
        self.inner_IJV = self.solver.lhs_sps_pde_inner(A=1, B=0, C=1,
                                                       D=h[0], E=h[1], F=0)
        init_lhs_sps = self.solver.assemble_lhs_sps(self.inner_IJV, self.bound_IJV)

        g = g0 + g1 * self.growth_func.get_recent_f()
        init_rhs = self.solver.assemble_rhs(g)

        p_sim = spsolve(init_lhs_sps, init_rhs).reshape(sim_shape)
        return self.solver.sim_result_to_sol(p_sim)


class IterativePressureSolver(PressureSolver):
    """
    del2(p) + h*grad(p) = g0 + g1 * lambda(p)
    LHS = del2 p1 + h * grad p1 - g1 * d_dp_lambda(p0) * p1
    RHS = g0 + g1 * lambda(p0) - g1 * d_dp_lambda(p0) * p0
    """

    def __init__(self, tol=1e-6, max_iter=20, **kwargs):
        super(IterativePressureSolver, self).__init__(**kwargs)
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, h: VectorField2D,
              g0: ScalarField2D,
              g1: ScalarField2D,
              p0: ScalarField2D = None,
              *args, **kwargs):
        # p0作为迭代的起始解
        if p0 is None:
            raise ValueError("Initial guess p0 can not be None")

        diff = 1
        n_iter = 0
        while diff > self.tol:
            if n_iter >= self.max_iter:
                msg = f'Iteration do not converge after {self.max_iter} steps. '
                msg += f'Relative error = {diff:.4e} > tol = {self.tol:.4e}'
                RuntimeLogging().error(msg)
                raise RuntimeError

            p1 = self._single_iterative(p0, h, g0, g1)
            diff = np.max(np.abs(p1 - p0))
            n_iter += 1
            p0 = p1

        return p0

    def _single_iterative(self, p0: ScalarField2D,
                          h: VectorField2D,
                          g0: ScalarField2D,
                          g1: ScalarField2D):
        if TYPE_CHECKING:
            self.growth_func: "PressureConstrainedGrowth"

        jacobian = -self.growth_func.d_dp_growth(p0) * g1
        self.inner_IJV = self.solver.lhs_sps_pde_inner(A=1, B=0, C=1,
                                                       D=h[0], E=h[1], F=jacobian)
        lhs_sps = self.solver.assemble_lhs_sps(self.inner_IJV, self.bound_IJV)

        lmbda_0 = self.growth_func.recent_full_f * self.growth_func.constrain_func.f(p0)
        g = g0 + g1 * lmbda_0 + jacobian * p0
        rhs = self.solver.assemble_rhs(g)

        sim_shape = self.solver.sim_shape
        p1_sim = spsolve(lhs_sps, rhs).reshape(sim_shape)
        return self.solver.sim_result_to_sol(p1_sim)

    def __str__(self):
        title_line = super().__str__() + "\n"
        info = f'tolerance = {self.tol:.4e} \n'
        info += f'max iter = {self.max_iter}'
        return title_line + info
