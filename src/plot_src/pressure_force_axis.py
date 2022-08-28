from typing import TYPE_CHECKING
import numpy as np

from .base_figure import BaseFigure
from src.tools.analyze_tools import Image
from src.tools.plot_tools import makeAnimation
from src.tools.num_tools import laplacian2D, grad2D

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class PressureForceAxis:
    def __init__(self, model, plot_theory=False):
        self.model: TwoFluidModel = model
        self.plot_theory = plot_theory
        self.fig = BaseFigure(fig_type="curve",
                              n_row_figs=2,
                              n_col_figs=1).fig
        self.x_grid = np.arange(model.N_COLUMN) * model.GRID
        self.y_grid = np.arange(model.N_ROW) * model.GRID

    def _p_theory(self):
        phi_binary_img = Image(self.model.phi).get_binary_image(self.model.PHI_CELL - 0.2, 1)
        cell_x_width = phi_binary_img.get_x_width()
        cell_y_width = self.model.N_ROW
        growth_eff = np.sum(self.model.growth) / (cell_x_width * cell_y_width)
        interface = self.model.GRID * cell_x_width

        v_ecm = interface * growth_eff
        p_profile = np.where(self.x_grid < interface,
                             -0.5 * self.model.ZETA_CELL * growth_eff * (self.x_grid ** 2),
                             -0.5 * self.model.ZETA_CELL * growth_eff * (interface ** 2)
                             - v_ecm * self.model.ZETA_ECM * (self.x_grid - interface))
        p_shift = np.min(p_profile)
        return p_profile - p_shift

    def _draw_single_frame(self, n_frame):
        self.model.load_frame(n_frame)
        self.fig.clear()
        ax_pressure = self.fig.add_subplot(211)
        ax_force = self.fig.add_subplot(212)

        row_idx = self.model.N_ROW // 2

        osmotic = - self.model.NRG * self.model.free_nrg.f(self.model.phi) \
                  + self.model.NRG * self.model.phi * self.model.free_nrg.df(self.model.phi)

        ax_pressure.plot(self.x_grid, self.model.pressure[row_idx, :], label="p")
        ax_pressure.plot(self.x_grid, osmotic[row_idx, :],
                         label=r'$\phi \frac{\partial f_{FH}}{\partial \phi} - f_{FH}$')
        ax_pressure.plot(self.x_grid, self.model.pressure[row_idx, :] + osmotic[row_idx, :],
                         label=r'$p_{tot}$')

        if self.plot_theory:
            p_theory = self._p_theory()
            ax_pressure.plot(self.x_grid, p_theory, label=r'$p_{theory}$')

        ax_pressure.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        ax_pressure.set_title(f"pressure_axis, "
                              f"T={self.model.sim_time:.2E}, "
                              f"Δx={self.model.view_moved * self.model.GRID:.2E}")

        del2_phi = laplacian2D(self.model.phi,
                               step_size=self.model.GRID,
                               boundary_x="truncate",
                               boundary_y="truncate")
        surf = -self.model.SURF * self.model.phi * grad2D(del2_phi,
                                                          step_size=self.model.GRID,
                                                          boundary_x="truncate",
                                                          boundary_y="truncate")
        osmotic_force = grad2D(osmotic,
                               step_size=self.model.GRID,
                               boundary_x="truncate",
                               boundary_y="truncate")
        ax_force.plot(self.x_grid, self.model.grad_p[0, row_idx, :], label=r'$\nabla p$', color='#00FFFF')
        ax_force.plot(self.x_grid, surf[0, row_idx, :],
                      label=r'$-\nabla (C \nabla^2 \phi)$',
                      color='#008080')
        ax_force.plot(self.x_grid, osmotic_force[0, row_idx, :],
                      label=r'$\phi \nabla (\frac{\partial f_{FH}}{\partial \phi})$',
                      color='#007FFF')
        ax_force.plot(self.x_grid,
                      (self.model.zeta * self.model.v_avg)[0, row_idx, :],
                      label=r'$\zeta (\phi) v_{avg}$',
                      color='#C3B091')
        ax_force.plot(self.x_grid,
                      (self.model.XI * (self.model.v_cell - self.model.v_avg)
                       / (1 - self.model.phi))[0, row_idx, :],
                      label=r'$\xi (v_c -v_e)$',
                      color='#F0E68C')
        ax_force.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        ax_force.set_title(f"force_axis, "
                           f"T={self.model.sim_time:.2E}, "
                           f"Δx={self.model.view_moved * self.model.GRID:.2E}")

    def run(self):
        total_frame = len(self.model.sim_time_frame)
        makeAnimation(self.fig, self._draw_single_frame, total_frame,
                      file_name=self.__class__.__name__)
