import numpy as np

from src.tools.pde_tools import OperatorLib, BoundName, BoundType
from src.tools.analyze_tools import BinaryImage
from src.tools.type_tools import ScalarField2D
from .base_model import BaseModel


class ViewMove(BaseModel):
    """
    在x方向上追踪tissue, 目前模拟中只会用到上一步迭代的phi和p
    view move只需要更新这两个量
    """

    def __init__(self, phi_bound, p_bound):
        super().__init__()
        self.move_direction = 0
        self.phi_bound: OperatorLib = phi_bound
        self.p_bound: OperatorLib = p_bound

    def set_move_direction(self, mat: ScalarField2D):
        raise NotImplementedError

    def phi_view_move(self, phi: ScalarField2D) -> ScalarField2D:
        raise NotImplementedError

    def p_view_move(self, pressure: ScalarField2D) -> ScalarField2D:
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__:=^40}'


class OffViewMove(ViewMove):

    def set_move_direction(self, mat: ScalarField2D):
        pass

    def phi_view_move(self, phi: ScalarField2D) -> ScalarField2D:
        return phi

    def p_view_move(self, pressure: ScalarField2D) -> ScalarField2D:
        return pressure


class HoldTissueCenter(ViewMove):

    def __init__(self, phi_bound, p_bound, tol=5):
        super().__init__(phi_bound, p_bound)
        self.init_x_mid = None
        self.tol = tol

    def get_tissue_x_mid(self, phi):
        img = BinaryImage(phi, self.PHI_CELL - 0.1, 1)
        temp_x_mid, _ = img.get_shape_center()
        return temp_x_mid

    def set_move_direction(self, mat: ScalarField2D):
        temp_x_mid = self.get_tissue_x_mid(phi=mat)
        if self.init_x_mid is None:
            self.init_x_mid = temp_x_mid

        if temp_x_mid > self.init_x_mid + self.tol:
            self.move_direction = 1  # move frame to right
        elif temp_x_mid < self.init_x_mid - self.tol:
            self.move_direction = -1  # move frame to left
        else:
            self.move_direction = 0

    def phi_view_move(self, phi: ScalarField2D) -> ScalarField2D:
        if self.move_direction == 0:
            return phi

        result = np.roll(phi, -self.move_direction, axis=1)
        if self.phi_bound.bound_condition[BoundName.LEFT] == BoundType.PERIODIC:
            return result

        else:
            if self.move_direction > 0:
                b_name = BoundName.RIGHT
                b_type = self.phi_bound.bound_condition[b_name]
                if b_type == BoundType.DIRICHLET:
                    result[:, -1] = self.phi_bound.bound_value[b_name]
                elif b_type == BoundType.NEUMANN:
                    delta = 2 * self.GRID * self.phi_bound.bound_value[b_name]
                    result[:, -1] = phi[:, -2] + delta
            elif self.move_direction < 0:
                b_name = BoundName.LEFT
                b_type = self.phi_bound.bound_condition[b_name]
                if b_type == BoundType.DIRICHLET:
                    result[:, 0] = self.phi_bound.bound_value[b_name]
                elif b_type == BoundType.NEUMANN:
                    delta = 2 * self.GRID * self.phi_bound.bound_value[b_name]
                    result[:, 0] = phi[:, 1] - delta
            return result

    def p_view_move(self, pressure: ScalarField2D) -> ScalarField2D:
        if self.move_direction == 0:
            return pressure

        result = np.roll(pressure, -self.move_direction, axis=1)
        if self.p_bound.bound_condition[BoundName.LEFT] == BoundType.PERIODIC:
            return result

        else:
            if self.move_direction > 0:
                b_name = BoundName.RIGHT
                b_type = self.p_bound.bound_condition[b_name]
                if b_type == BoundType.DIRICHLET:
                    result[:, -1] = self.p_bound.bound_value[b_name]
                elif b_type == BoundType.NEUMANN:
                    delta = 2 * self.GRID * self.p_bound.bound_value[b_name]
                    result[:, -1] = pressure[:, -2] + delta
            elif self.move_direction < 0:
                b_name = BoundName.LEFT
                b_type = self.p_bound.bound_condition[b_name]
                if b_type == BoundType.DIRICHLET:
                    result[:, 0] = self.p_bound.bound_value[b_name]
                elif b_type == BoundType.NEUMANN:
                    delta = 2 * self.GRID * self.p_bound.bound_value[b_name]
                    result[:, 0] = pressure[:, 1] - delta
            return result

    def __str__(self):
        title_line = super().__str__() + "\n"
        expression = f'tolerance = {self.tol}'
        return title_line + expression
