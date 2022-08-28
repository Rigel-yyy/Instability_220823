# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:48:43 2022

@author: Rigel_yyy
"""

from .type_tools import ScalarField2D

import scipy.sparse as sps
import numpy as np
from scipy.ndimage import convolve1d
from collections import Counter
from typing import Tuple, Dict
from enum import Enum


class BoundName(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @classmethod
    def opposite(cls, name):
        match name:
            case BoundName.UP:
                return BoundName.DOWN
            case BoundName.DOWN:
                return BoundName.UP
            case BoundName.LEFT:
                return BoundName.RIGHT
            case BoundName.RIGHT:
                return BoundName.LEFT


class BoundType(Enum):
    PERIODIC = 0
    DIRICHLET = 1
    NEUMANN = 2


class BaseLatticeGrid:

    def __init__(self, shape: Tuple[int],
                 bound_type: Dict[BoundName, BoundType],
                 grid_size: float = 1.0):
        """
        Initialing a lattice grid size with specific size and boundary condition        

        Parameters
        ----------
        shape : Tuple[int]
            Size of the whole solution field
        .
        """

        self.bound_condition = bound_type
        self.sol_shape = shape
        self.grid_size = grid_size
        self._is_validate()
        self.sol_region, self.sim_region, self.inner_region = self.get_region()
        self.sim_shape = self.get_shape("sim")
        self.inner_shape = self.get_shape("inner")
        self.sim_bound_slice = self.get_sim_boundary_slice()
        self.sim_inner_slice = self.get_sim_inner_slice()
        self.sol_bound_slice = self.get_sol_boundary_slice()
        self.sol_inner_slice = self.get_sol_inner_slice()

    def _is_validate(self):
        """
        Periodic boundary condition should appear in pair
        """

        if self.bound_condition[BoundName.UP] == BoundType.PERIODIC:
            if self.bound_condition[BoundName.DOWN] != BoundType.PERIODIC:
                raise ValueError("Inconsistent up down boundary condition")

        if self.bound_condition[BoundName.DOWN] == BoundType.PERIODIC:
            if self.bound_condition[BoundName.UP] != BoundType.PERIODIC:
                raise ValueError("Inconsistent up down boundary condition")

        if self.bound_condition[BoundName.LEFT] == BoundType.PERIODIC:
            if self.bound_condition[BoundName.RIGHT] != BoundType.PERIODIC:
                raise ValueError("Inconsistent left right boundary condition")

        if self.bound_condition[BoundName.RIGHT] == BoundType.PERIODIC:
            if self.bound_condition[BoundName.LEFT] != BoundType.PERIODIC:
                raise ValueError("Inconsistent left right boundary condition")

    def get_region(self):
        """
        sol region is the mathematical region
        inner region satisfies PDE defined within
        simulation region includes the inner region and edge region
        """

        sol_row_idx = np.array([0, self.sol_shape[0] - 1])
        sol_col_idx = np.array([0, self.sol_shape[1] - 1])
        sim_row_idx = np.array([0, self.sol_shape[0] - 1])
        sim_col_idx = np.array([0, self.sol_shape[1] - 1])
        inner_row_idx = np.array([0, self.sol_shape[0] - 1])
        inner_col_idx = np.array([0, self.sol_shape[1] - 1])

        match self.bound_condition[BoundName.UP]:
            case BoundType.PERIODIC:
                sim_row_idx[1] -= 1
                inner_row_idx[1] -= 1
            case BoundType.DIRICHLET:
                inner_row_idx[1] -= 1
            case BoundType.NEUMANN:
                sim_row_idx[1] += 1

        match self.bound_condition[BoundName.DOWN]:
            case BoundType.PERIODIC:
                pass
            case BoundType.DIRICHLET:
                inner_row_idx[0] += 1
            case BoundType.NEUMANN:
                sim_row_idx[0] -= 1

        match self.bound_condition[BoundName.LEFT]:
            case BoundType.PERIODIC:
                pass
            case BoundType.DIRICHLET:
                inner_col_idx[0] += 1
            case BoundType.NEUMANN:
                sim_col_idx[0] -= 1

        match self.bound_condition[BoundName.RIGHT]:
            case BoundType.PERIODIC:
                sim_col_idx[1] -= 1
                inner_col_idx[1] -= 1
            case BoundType.DIRICHLET:
                inner_col_idx[1] -= 1
            case BoundType.NEUMANN:
                sim_col_idx[1] += 1

        # index < 0 is not allowed, shift minimal index to 0
        row_shift = -sim_row_idx[0]
        col_shift = -sim_col_idx[0]

        sol_row_idx += row_shift
        sim_row_idx += row_shift
        inner_row_idx += row_shift
        sol_col_idx += col_shift
        sim_col_idx += col_shift
        inner_col_idx += col_shift

        return ([sol_row_idx, sol_col_idx],
                [sim_row_idx, sim_col_idx],
                [inner_row_idx, inner_col_idx])

    def get_shape(self, region: str):
        """
        get shape of the region
        region : str {"sol"|"sim"|"inner"}
        """
        match region:
            case "sol":
                return (self.sol_region[0][1] - self.sol_region[0][0] + 1,
                        self.sol_region[1][1] - self.sol_region[1][0] + 1)
            case "sim":
                return (self.sim_region[0][1] - self.sim_region[0][0] + 1,
                        self.sim_region[1][1] - self.sim_region[1][0] + 1)
            case "inner":
                return (self.inner_region[0][1] - self.inner_region[0][0] + 1,
                        self.inner_region[1][1] - self.inner_region[1][0] + 1)
            case _:
                raise ValueError(f"No region name {region}")

    def _single_sim_boundary_slice(self, name, region: str,
                                   start: int = 0, end: int = 0):
        """
        get boundary slice of sim region
        --------------
        name : BoundName
        region : str {"sim"|"sol"|"inner"}
        start : int >= 0, start index
        end : int <= 0 , end index
        --------------
        return slice object for name
        """
        region_idx = None
        if region == "sim":
            region_idx = self.sim_region
        elif region == "sol":
            region_idx = self.sol_region
        elif region == "inner":
            region_idx = self.inner_region

        match name:
            case BoundName.UP:
                return np.s_[-1, region_idx[1][0] + start: region_idx[1][1] + 1 + end]
            case BoundName.DOWN:
                return np.s_[0, region_idx[1][0] + start: region_idx[1][1] + 1 + end]
            case BoundName.LEFT:
                return np.s_[region_idx[0][0] + start: region_idx[0][1] + 1 + end, 0]
            case BoundName.RIGHT:
                return np.s_[region_idx[0][0] + start: region_idx[0][1] + 1 + end, -1]

    @staticmethod
    def _single_sol_boundary_slice(name):
        """
        get boundary slice of sol region
        ---------
        name : BoundName
        ---------
        return slice object for name
        """
        match name:
            case BoundName.UP:
                return np.s_[-1, :]
            case BoundName.DOWN:
                return np.s_[0, :]
            case BoundName.LEFT:
                return np.s_[:, 0]
            case BoundName.RIGHT:
                return np.s_[:, -1]

    def get_sim_boundary_slice(self):
        """
        get slice of boundary from simulation region
        for periodic boundary None type returned
        """
        sim_bound_slice = {name: None for name in BoundName}
        n_condition = Counter(self.bound_condition.values())

        if n_condition[BoundType.PERIODIC] == 2:
            for b_name, b_type in self.bound_condition.items():
                if b_type in (BoundType.DIRICHLET, BoundType.NEUMANN):
                    sim_bound_slice[b_name] = \
                        self._single_sim_boundary_slice(b_name, "sim")

        elif n_condition[BoundType.PERIODIC] == 0:
            if n_condition[BoundType.DIRICHLET] == 0:
                for b_name in (BoundName.LEFT, BoundName.RIGHT):
                    sim_bound_slice[b_name] = \
                        self._single_sim_boundary_slice(b_name, "sim")
                for b_name in (BoundName.UP, BoundName.DOWN):
                    sim_bound_slice[b_name] = \
                        self._single_sim_boundary_slice(b_name, "inner")

            elif n_condition[BoundType.DIRICHLET] == 1:
                for b_name, b_type in self.bound_condition.items():
                    if b_type == BoundType.DIRICHLET:
                        ob_name = BoundName.opposite(b_name)
                        other_b_name = list(BoundName)
                        other_b_name.remove(b_name)
                        other_b_name.remove(ob_name)

                        if b_name in (BoundName.LEFT, BoundName.RIGHT):
                            sim_bound_slice[b_name] = \
                                self._single_sim_boundary_slice(b_name, "inner")
                            sim_bound_slice[ob_name] = \
                                self._single_sim_boundary_slice(ob_name, "sim")
                            for side in other_b_name:
                                sim_bound_slice[side] = \
                                    self._single_sim_boundary_slice(side, "sol")

                        else:
                            sim_bound_slice[b_name] = \
                                self._single_sim_boundary_slice(b_name, "inner")
                            sim_bound_slice[ob_name] = \
                                self._single_sim_boundary_slice(ob_name, "inner")
                            for side in other_b_name:
                                sim_bound_slice[side] = \
                                    self._single_sim_boundary_slice(side, "sim")

            elif n_condition[BoundType.DIRICHLET] == 2:
                # opposite boundary has same boundary type
                if (self.bound_condition[BoundName.UP]
                        == self.bound_condition[BoundName.DOWN]):
                    for b_name, b_type in self.bound_condition.items():
                        if b_type == BoundType.DIRICHLET:
                            sim_bound_slice[b_name] = \
                                self._single_sim_boundary_slice(b_name, "sol")
                        else:
                            sim_bound_slice[b_name] = \
                                self._single_sim_boundary_slice(b_name, "sim")
                else:
                    for b_name in (BoundName.LEFT, BoundName.RIGHT):
                        if self.bound_condition[b_name] == BoundType.DIRICHLET:
                            sim_bound_slice[b_name] = \
                                self._single_sim_boundary_slice(b_name, "inner")
                        else:
                            sim_bound_slice[b_name] = \
                                self._single_sim_boundary_slice(b_name, "sim")
                    for b_name in (BoundName.UP, BoundName.DOWN):
                        sim_bound_slice[b_name] = \
                            self._single_sim_boundary_slice(b_name, "sol")

            elif n_condition[BoundType.DIRICHLET] == 3:
                for b_name, b_type in self.bound_condition.items():
                    if b_type == BoundType.NEUMANN:
                        ob_name = BoundName.opposite(b_name)
                        other_b_name = list(BoundName)
                        other_b_name.remove(b_name)
                        other_b_name.remove(ob_name)

                        sim_bound_slice[b_name] = \
                            self._single_sim_boundary_slice(b_name, "sim")
                        sim_bound_slice[ob_name] = \
                            self._single_sim_boundary_slice(ob_name, "inner")

                        for side in other_b_name:
                            sim_bound_slice[side] = \
                                self._single_sim_boundary_slice(side, "sol")

            elif n_condition[BoundType.DIRICHLET] == 4:
                for b_name in (BoundName.UP, BoundName.DOWN):
                    sim_bound_slice[b_name] = \
                        self._single_sim_boundary_slice(b_name, "inner")
                for b_name in (BoundName.LEFT, BoundName.RIGHT):
                    sim_bound_slice[b_name] = \
                        self._single_sim_boundary_slice(b_name, "sim")

        return sim_bound_slice

    def get_sim_inner_slice(self):
        """
        get slice object for inner regions from sim region
        """

        row_s = self.inner_region[0][0]
        row_e = self.inner_region[0][1] + 1
        col_s = self.inner_region[1][0]
        col_e = self.inner_region[1][1] + 1

        return np.s_[row_s:row_e, col_s:col_e]

    def get_sol_boundary_slice(self):

        sol_bound_slice = {name: None for name in BoundName}
        for b_name, b_type in self.bound_condition.items():
            if b_type != BoundType.NEUMANN:
                sol_bound_slice[b_name] = self._single_sol_boundary_slice(b_name)
        return sol_bound_slice

    def get_sol_inner_slice(self):
        """
        get slice object for inner regions from sol region
        """

        row_s = self.inner_region[0][0] - self.sol_region[0][0]
        row_e = self.inner_region[0][1] - self.sol_region[0][0] + 1
        col_s = self.inner_region[1][0] - self.sol_region[1][0]
        col_e = self.inner_region[1][1] - self.sol_region[1][0] + 1

        return np.s_[row_s:row_e, col_s:col_e]


class OperatorLib(BaseLatticeGrid):
    """
    标量场在特定边界条件下的梯度，拉普拉斯运算
    """

    def __init__(self, bound_value, **kwargs):
        super().__init__(**kwargs)
        self.sim_neumann_src_slice = self.get_sim_neumann_source_slice()

        self.bound_value = None
        self.neumann_diff_value = None
        self.set_bound_value(bound_value)

    def set_bound_value(self, bound_value):
        self.bound_value = bound_value
        self.check_bound_value()
        self.neumann_diff_value = self.get_neumann_diff_value()

    def check_bound_value(self):
        for b_name, b_type in self.bound_condition.items():
            if b_type in (BoundType.DIRICHLET, BoundType.NEUMANN):
                if self.bound_value[b_name] is None:
                    raise ValueError(f"Wrong boundary value for {b_name}.")

    def get_sim_neumann_source_slice(self):
        """
        扩充sim region时，Neumann边界条件要用到内部的值
        """
        src_slice = {b_name: None for b_name in BoundName}

        for b_name, b_type in self.bound_condition.items():
            if b_type == BoundType.NEUMANN:
                des_slice = self.sim_bound_slice[b_name]
                match b_name:
                    case BoundName.UP:
                        src_slice[b_name] = (des_slice[0] - 2, des_slice[1])
                    case BoundName.DOWN:
                        src_slice[b_name] = (des_slice[0] + 2, des_slice[1])
                    case BoundName.LEFT:
                        src_slice[b_name] = (des_slice[0], des_slice[1] + 2)
                    case BoundName.RIGHT:
                        src_slice[b_name] = (des_slice[0], des_slice[1] - 2)

        return src_slice

    def get_neumann_diff_value(self):
        """
        get difference value on Neumann boundary
        diff_value[BoundaryName] = des_value - src_value
        """
        diff_value = {b_name: None for b_name in BoundName}

        for b_name, b_type in self.bound_condition.items():
            if b_type == BoundType.NEUMANN:
                coeff = 2 * self.grid_size
                match b_name:
                    case BoundName.UP:
                        diff_value[b_name] = self.bound_value[b_name] * coeff
                    case BoundName.DOWN:
                        diff_value[b_name] = -self.bound_value[b_name] * coeff
                    case BoundName.LEFT:
                        diff_value[b_name] = -self.bound_value[b_name] * coeff
                    case BoundName.RIGHT:
                        diff_value[b_name] = self.bound_value[b_name] * coeff

        return diff_value

    def expand_inner_to_sol(self, mat: ScalarField2D):
        """
        mat: field of inner region
        """
        result = np.zeros(self.sol_shape)
        result[self.sol_inner_slice] = mat

        # extend up and right periodic boundary
        for b_name, b_type in self.bound_condition.items():
            if b_type == BoundType.PERIODIC:
                if b_name in (BoundName.UP, BoundName.RIGHT):
                    ob_name = BoundName.opposite(b_name)
                    b_slice = self.sol_bound_slice[b_name]
                    ob_slice = self.sol_bound_slice[ob_name]
                    result[b_slice] = result[ob_slice]

        # extend all dirichlet boundary
        for b_name, b_type in self.bound_condition.items():
            if b_type == BoundType.DIRICHLET:
                b_slice = self.sol_bound_slice[b_name]
                result[b_slice] = self.bound_value[b_name]

        return result

    def expand_inner_to_sim(self, mat: ScalarField2D):
        """
        mat: field of inner region
        """
        result = np.zeros(self.sim_shape)
        result[self.sim_inner_slice] = mat

        # extend all dirichlet boundary
        for b_name, b_type in self.bound_condition.items():
            if b_type == BoundType.DIRICHLET:
                b_slice = self.sim_bound_slice[b_name]
                result[b_slice] = self.bound_value[b_name]

        # extend all neumann boundary
        for b_name, b_type in self.bound_condition.items():
            if b_type == BoundType.NEUMANN:
                des_slice = self.sim_bound_slice[b_name]
                src_slice = self.sim_neumann_src_slice[b_name]
                diff = self.neumann_diff_value[b_name]

                result[des_slice] = result[src_slice] + diff

        return result

    def d_dx(self, mat: ScalarField2D):
        """
        mat: field of sol region
        """

        if self.bound_condition[BoundName.LEFT] == BoundType.PERIODIC:
            stencil = np.array([1.0, 0.0, -1.0]) / (2 * self.grid_size)
            result = np.zeros(mat.shape)
            result[:, 0:-1] = convolve1d(mat[:, 0:-1].astype(float),
                                         stencil, axis=1, mode='wrap')
            result[:, -1] = result[:, 0]
            return result

        else:
            result = np.gradient(mat, axis=1, edge_order=2) / self.grid_size
            if self.bound_condition[BoundName.LEFT] == BoundType.NEUMANN:
                result[:, 0] = self.bound_value[BoundName.LEFT]

            if self.bound_condition[BoundName.RIGHT] == BoundType.NEUMANN:
                result[:, -1] = self.bound_value[BoundName.RIGHT]
            return result

    def d2_dx2(self, mat: ScalarField2D):
        result = np.zeros(mat.shape)
        dx = self.grid_size
        stencil = np.array([1.0, -2.0, 1.0]) / (dx**2)
        if self.bound_condition[BoundName.LEFT] == BoundType.PERIODIC:
            result[:, 0:-1] = convolve1d(mat[:, 0:-1].astype(float),
                                         stencil, axis=1, mode='wrap')
            result[:, -1] = result[:, 0]
            return result

        else:
            result = convolve1d(mat.astype(float),
                                stencil, axis=1, mode='wrap')
            if self.bound_condition[BoundName.LEFT] == BoundType.NEUMANN:
                bound_d_dx = self.bound_value[BoundName.LEFT]
                result[:, 0] = 2 * (mat[:, 1] - mat[:, 0] - dx * bound_d_dx) / dx**2
            elif self.bound_condition[BoundName.LEFT] == BoundType.DIRICHLET:
                result[:, 0] = result[:, 1]

            if self.bound_condition[BoundName.RIGHT] == BoundType.NEUMANN:
                bound_d_dx = self.bound_value[BoundName.RIGHT]
                result[:, -1] = 2 * (mat[:, -2] - mat[:, -1] + dx * bound_d_dx) / dx**2
            elif self.bound_condition[BoundName.RIGHT] == BoundType.DIRICHLET:
                result[:, -1] = result[:, -2]

            return result

    def d_dy(self, mat: ScalarField2D):
        """
        mat: field of sol region
        """

        if self.bound_condition[BoundName.UP] == BoundType.PERIODIC:
            stencil = np.array([1.0, 0.0, -1.0]) / (2 * self.grid_size)
            result = np.zeros(mat.shape)
            result[0:-1, :] = convolve1d(mat[0:-1, :].astype(float),
                                         stencil, axis=0, mode='wrap')
            result[-1, :] = result[0, :]
            return result
        else:
            result = np.gradient(mat, axis=0, edge_order=2) / self.grid_size
            if self.bound_condition[BoundName.UP] == BoundType.NEUMANN:
                result[-1, :] = self.bound_value[BoundName.UP]

            if self.bound_condition[BoundName.DOWN] == BoundType.NEUMANN:
                result[0, :] = self.bound_value[BoundName.DOWN]
            return result

    def d2_dy2(self, mat: ScalarField2D):
        dx = self.grid_size
        result = np.zeros(mat.shape)
        stencil = np.array([1.0, -2.0, 1.0]) / (dx**2)
        if self.bound_condition[BoundName.UP] == BoundType.PERIODIC:
            result[:, 0:-1] = convolve1d(mat[:, 0:-1].astype(float),
                                         stencil, axis=0, mode='wrap')
            result[:, -1] = result[:, 0]
            return result

        else:
            result = convolve1d(mat.astype(float), stencil,
                                axis=0, mode='wrap')
            if self.bound_condition[BoundName.UP] == BoundType.NEUMANN:
                bound_d_dy = self.bound_value[BoundName.UP]
                result[-1, :] = 2 * (mat[-2, :] - mat[-1, :] + bound_d_dy * dx) / dx**2
            elif self.bound_condition[BoundName.UP] == BoundType.DIRICHLET:
                result[-1, :] = result[-2, :]

            if self.bound_condition[BoundName.DOWN] == BoundType.NEUMANN:
                bound_d_dy = self.bound_value[BoundName.DOWN]
                result[0, :] = 2 * (mat[1, :] - mat[0, :] - bound_d_dy * dx) / dx**2
            elif self.bound_condition[BoundName.DOWN] == BoundType.DIRICHLET:
                result[0, :] = result[1, :]

            return result

    def grad(self, mat: ScalarField2D):
        """
        mat: field of sol region
        """
        return np.array([self.d_dx(mat), self.d_dy(mat)])

    def laplace(self, mat: ScalarField2D):
        return self.d2_dx2(mat) + self.d2_dy2(mat)


class Linear2ndPDESolver(OperatorLib):

    def __init__(self, stencil: {5 or 9} = 5, **kwargs):
        super().__init__(**kwargs)
        self.stencil_order = stencil
        self.neighbour_slice = self.get_neighbour_slice()
        self.lhs_sps = None
    
    def get_neighbour_slice(self):
        """
        get slice of the inner neighbourhood region from simulation region
        return Dict[direction : numpy slice]
        """
        neighbour_s_ = {}
        inner = self.inner_region
        sim = self.sim_region
        
        row_s = inner[0][0]
        row_e = inner[0][1] + 1
        col_s = inner[1][0]
        col_e = inner[1][1] + 1

        # update ld, cd, rd
        if inner[0][0] != sim[0][0]:
            neighbour_s_["cd"] = np.s_[row_s - 1:row_e - 1, col_s:col_e]
            if inner[1][0] != sim[1][0]:
                neighbour_s_["ld"] = np.s_[row_s - 1:row_e - 1, col_s - 1:col_e - 1]
            else:
                col_ids = np.arange(col_s - 1, col_e - 1)
                neighbour_s_["ld"] = np.s_[row_s - 1:row_e - 1, col_ids]

            if inner[1][1] != sim[1][1]:
                neighbour_s_["rd"] = np.s_[row_s - 1:row_e - 1, col_s + 1:col_e + 1]
            else:
                col_ids = np.arange(col_s + 1, col_e + 1)
                col_ids[-1] = 0
                neighbour_s_["rd"] = np.s_[row_s - 1:row_e - 1, col_ids]
        else:
            row_ids = np.arange(row_s - 1, row_e - 1)
            row_ids = row_ids[:, np.newaxis]
            neighbour_s_["cd"] = np.s_[row_ids.flatten(), col_s:col_e]

            if inner[1][0] != sim[1][0]:
                neighbour_s_["ld"] = np.s_[row_ids.flatten(), col_s - 1:col_e - 1]
            else:
                col_ids = np.arange(col_s - 1, col_e - 1)
                neighbour_s_["ld"] = np.s_[row_ids, col_ids]

            if inner[1][1] != inner[1][1]:
                neighbour_s_["rd"] = np.s_[row_ids.flatten(), col_s + 1:col_e + 1]
            else:
                col_ids = np.arange(col_s + 1, col_e + 1)
                col_ids[-1] = 0
                neighbour_s_["rd"] = np.s_[row_ids, col_ids]

        # update lu, cu, ru
        if inner[0][1] != sim[0][1]:
            neighbour_s_["cu"] = np.s_[row_s + 1:row_e + 1, col_s:col_e]
            if inner[1][0] != sim[1][0]:
                neighbour_s_["lu"] = np.s_[row_s + 1:row_e + 1, col_s - 1:col_e - 1]
            else:
                col_ids = np.arange(col_s - 1, col_e - 1)
                neighbour_s_["lu"] = np.s_[row_s + 1:row_e + 1, col_ids]

            if inner[1][1] != sim[1][1]:
                neighbour_s_["ru"] = np.s_[row_s + 1:row_e + 1, col_s + 1:col_e + 1]
            else:
                col_ids = np.arange(col_s + 1, col_e + 1)
                col_ids[-1] = 0
                neighbour_s_["ru"] = np.s_[row_s + 1:row_e + 1, col_ids]
        else:
            row_ids = np.arange(row_s + 1, row_e + 1)
            row_ids[-1] = 0
            row_ids = row_ids[:, np.newaxis]
            neighbour_s_["cu"] = np.s_[row_ids.flatten(), col_s:col_e]
            if inner[1][0] != sim[1][0]:
                neighbour_s_["lu"] = np.s_[row_ids.flatten(), col_s - 1:col_e - 1]
            else:
                col_ids = np.arange(col_s - 1, col_e - 1)
                neighbour_s_["lu"] = np.s_[row_ids, col_ids]

            if inner[1][1] != sim[1][1]:
                neighbour_s_["ru"] = np.s_[row_ids.flatten(), col_s + 1:col_e + 1]
            else:
                col_ids = np.arange(col_s + 1, col_e + 1)
                col_ids[-1] = 0
                neighbour_s_["ru"] = np.s_[row_ids, col_ids]

        # update cl, cr
        if inner[1][0] != sim[1][0]:
            neighbour_s_["cl"] = np.s_[row_s:row_e, col_s - 1:col_e - 1]
        else:
            col_ids = np.arange(col_s - 1, col_e - 1)
            neighbour_s_["cl"] = np.s_[row_s:row_e, col_ids]

        if inner[1][1] != sim[1][1]:
            neighbour_s_["cr"] = np.s_[row_s:row_e, col_s + 1:col_e + 1]
        else:
            col_ids = np.arange(col_s + 1, col_e + 1)
            col_ids[-1] = 0
            neighbour_s_["cr"] = np.s_[row_s:row_e, col_ids]

        return neighbour_s_

    def check_stencil_order(self,
                            A: {float or ScalarField2D},
                            C: {float or ScalarField2D}):
        if self.stencil_order == 5:
            return

        # 9 point stencil only applies to laplacian operator which means A=C
        if isinstance(A, (int, float)) and isinstance(C, (int, float)):
            if A != C:
                self.stencil_order = 5
        else:
            self.stencil_order = 5

    def _to_inner_flatten(self, mat: {float or ScalarField2D}):
        """
        isinstance(mat, ScalarField2D): extract inner region from mat with sol shape
        isinstance(mat, float): expand mat to inner shape
        ---------
        return flattened array
        """

        if isinstance(mat, np.ndarray):
            return mat[self.sol_inner_slice].flatten()
        else:
            idd = np.ones(np.prod(self.inner_shape))
            return idd * mat

    def lhs_sps_pde_inner(self,
                             A: {float or ScalarField2D} = 0,
                             B: {float or ScalarField2D} = 0,
                             C: {float or ScalarField2D} = 0,
                             D: {float or ScalarField2D} = 0,
                             E: {float or ScalarField2D} = 0,
                             F: {float or ScalarField2D} = 0):
        """
        sparse matrix representation of a 2nd order linear PDE
        Au_xx + Bu_xy + Cu_yy + Du_x + Eu_y + Fu

        Note that boundary grids should be dealt with additionally
        如果输入的是矩阵，它应该具有sol_region的大小

        Parameters
        ----------
        A : {float or ScalarField2D}. The default is 0.
        B : {float or ScalarField2D}. The default is 0.
        C : {float or ScalarField2D}. The default is 0.
        D : {float or ScalarField2D}. The default is 0.
        E : {float or ScalarField2D}. The default is 0.
        F : {float or ScalarField2D}. The default is 0.

        Returns
        -------
        return I, J, V for coo_sparse matrix construction

        """

        d = 1 / self.grid_size

        full_size = np.prod(self.sim_shape)
        sps_idx = np.arange(full_size).reshape(self.sim_shape)

        A, C, D, E, F = [self._to_inner_flatten(item) for item in (A, C, D, E, F)]

        I_cc = sps_idx[self.sim_inner_slice].flatten()

        self.check_stencil_order(A, C)

        J_cl = sps_idx[self.neighbour_slice["cl"]].flatten()
        J_cr = sps_idx[self.neighbour_slice["cr"]].flatten()
        J_cu = sps_idx[self.neighbour_slice["cu"]].flatten()
        J_cd = sps_idx[self.neighbour_slice["cd"]].flatten()
        J_ld = sps_idx[self.neighbour_slice["ld"]].flatten()
        J_rd = sps_idx[self.neighbour_slice["rd"]].flatten()
        J_lu = sps_idx[self.neighbour_slice["lu"]].flatten()
        J_ru = sps_idx[self.neighbour_slice["ru"]].flatten()

        if self.stencil_order == 5:
            V_cl = A * d ** 2 - 0.5 * D * d
            V_cr = A * d ** 2 + 0.5 * D * d
            V_cu = C * d ** 2 + 0.5 * E * d
            V_cd = C * d ** 2 - 0.5 * E * d
            V_cc = -2 * (A + C) * d ** 2 + F

            I = np.concatenate((I_cc, I_cc, I_cc, I_cc, I_cc))
            J = np.concatenate((J_cl, J_cd, I_cc, J_cu, J_cr))
            V = np.concatenate((V_cl, V_cd, V_cc, V_cu, V_cr))

            if isinstance(B, (int or float)) and B == 0:
                return I, J, V
            else:
                B = self._to_inner_flatten(B)

                V_ld = 0.25 * B * d ** 2
                V_rd = -0.25 * B * d ** 2
                V_lu = -0.25 * B * d ** 2
                V_ru = 0.25 * B * d ** 2
                I = np.concatenate((I, I_cc, I_cc, I_cc, I_cc))
                J = np.concatenate((J, J_ld, J_rd, J_lu, J_ru))
                V = np.concatenate((V, V_ld, V_rd, V_lu, V_ru))
                return I, J, V
        else:
            B = self._to_inner_flatten(B)

            V_cl = 2 / 3 * A * d ** 2 - 0.5 * D * d
            V_cr = 2 / 3 * A * d ** 2 + 0.5 * D * d
            V_cu = 2 / 3 * A * d ** 2 + 0.5 * E * d
            V_cd = 2 / 3 * A * d ** 2 - 0.5 * E * d
            V_cc = -10 / 3 * A * d ** 2 + F
            V_ld = (A / 6 + 0.25 * B) * d ** 2
            V_rd = (A / 6 - 0.25 * B) * d ** 2
            V_lu = (A / 6 - 0.25 * B) * d ** 2
            V_ru = (A / 6 + 0.25 * B) * d ** 2

            I = np.concatenate((I_cc, I_cc, I_cc, I_cc, I_cc,
                                I_cc, I_cc, I_cc, I_cc))
            J = np.concatenate((J_cl, J_cd, I_cc, J_cu, J_cr,
                                J_ld, J_rd, J_lu, J_ru))
            V = np.concatenate((V_cl, V_cd, V_cc, V_cu, V_cr,
                                V_ld, V_rd, V_lu, V_ru))
            return I, J, V

    def lhs_sps_pde_boundary(self):
        """
        sparse coefficients for boundary condition
        generate I,J,V array for COO_sparse matrix creation
        """
        d = 1 / self.grid_size
        nrow, ncol = self.sim_shape
        full_size = nrow * ncol
        sps_idx = np.arange(full_size).reshape((nrow, ncol))

        bond_slice = self.sim_bound_slice

        I, J, V = [], [], []
        for b_name, b_type in self.bound_condition.items():
            I_cc = sps_idx[bond_slice[b_name]]
            idd = np.ones(I_cc.size)
            if b_type == BoundType.DIRICHLET:
                I.append(I_cc)
                J.append(I_cc)
                V.append(idd)
            elif b_type == BoundType.NEUMANN:
                match b_name:
                    case BoundName.UP:
                        I.append(I_cc)
                        J.append(I_cc)
                        V.append(0.5 * idd * d)
                        I.append(I_cc)
                        J.append(I_cc - 2 * ncol)
                        V.append(-0.5 * idd * d)
                    case BoundName.DOWN:
                        I.append(I_cc)
                        J.append(I_cc)
                        V.append(-0.5 * idd * d)
                        I.append(I_cc)
                        J.append(I_cc + 2 * ncol)
                        V.append(0.5 * idd * d)
                    case BoundName.LEFT:
                        I.append(I_cc)
                        J.append(I_cc)
                        V.append(-0.5 * idd * d)
                        I.append(I_cc)
                        J.append(I_cc + 2)
                        V.append(0.5 * idd * d)
                    case BoundName.RIGHT:
                        I.append(I_cc)
                        J.append(I_cc)
                        V.append(0.5 * idd * d)
                        I.append(I_cc)
                        J.append(I_cc - 2)
                        V.append(-0.5 * idd * d)

        return np.concatenate(I), np.concatenate(J), np.concatenate(V)

    def assemble_lhs_sps(self, inner_IJV, bound_IJV):
        """
        内部和边界都用I J V格式记录的COO稀疏矩阵
        对应的是求解区域sim_region的系数
        组装成CSR格式的矩阵并输出
        """

        I_inner, J_inner, V_inner = inner_IJV
        I_bound, J_bound, V_bound = bound_IJV
        coeff_size = np.prod(self.sim_shape)
        I = np.concatenate((I_bound, I_inner))
        J = np.concatenate((J_bound, J_inner))
        V = np.concatenate((V_bound, V_inner))
        self.lhs_sps = sps.coo_matrix((V, (I, J)),
                                      shape=(coeff_size, coeff_size),
                                      dtype=float).tocsr()
        return self.lhs_sps

    def assemble_rhs(self, mat: {float or ScalarField2D} = 0):
        """
        输入RHS mat应为一个数或是具有sol_region大小的矩阵，
        返回具有sim_region大小被压扁后的一维数组

        Parameters
        ------------
        mat: {float or ScalarField2D} = 0

        """
        sim_rhs = np.zeros(self.sim_shape)
        if isinstance(mat, float):
            sim_rhs[self.sim_inner_slice] = mat
        else:
            sim_rhs[self.sim_inner_slice] = mat[self.sol_inner_slice]

        for b_name, b_type in self.bound_condition.items():
            if b_type != BoundType.PERIODIC:
                b_slice = self.sim_bound_slice[b_name]
                sim_rhs[b_slice] = self.bound_value[b_name]

        return sim_rhs.flatten()

    def sim_result_to_sol(self, mat: ScalarField2D) -> ScalarField2D:
        """
        将方程的解从sim_region转换到sol_region
        """
        inner_mat = mat[self.sim_inner_slice]
        return self.expand_inner_to_sol(inner_mat)
