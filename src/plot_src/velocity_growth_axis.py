from typing import TYPE_CHECKING
import numpy as np

from .base_figure import BaseFigure
from src.tools.analyze_tools import Image
from src.tools.plot_tools import makeAnimation
from src.tools.num_tools import div2D
from src.model.active_growth import PressureConstrainedGrowth

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class VelocityGrowthAxis:
    def __init__(self, model, plot_theory=False):
        self.model: TwoFluidModel = model
        self.plot_theory = plot_theory
        self.fig = BaseFigure(fig_type="curve",
                              n_row_figs=2,
                              n_col_figs=1).fig
        self.x_grid = np.arange(model.N_COLUMN) * model.GRID
        self.y_grid = np.arange(model.N_ROW) * model.GRID

    def _v_theory(self):
        phi_binary_img = Image(self.model.phi).\
            get_binary_image(self.model.PHI_CELL-0.005, 1)
        cell_x_width = phi_binary_img.get_x_width()
        cell_y_width = self.model.N_ROW
        growth_eff = np.sum(self.model.growth) / (cell_x_width * cell_y_width)
        interface = self.model.GRID * cell_x_width

        v_profile = np.where(self.x_grid < interface,
                             growth_eff * self.x_grid,
                             growth_eff * interface)
        return v_profile

    def _draw_single_frame(self, n_frame):
        self.model.load_frame(n_frame)
        self.fig.clear()

        ax1_phi = self.fig.add_subplot(211)
        ax2_velocity = self.fig.add_subplot(212)

        des_row = self.model.N_ROW // 2
        ax1_phi.set_title(f"phi, "
                          f"T={self.model.sim_time:.2E}, "
                          f"Δx={self.model.view_moved * self.model.GRID:.2E}")

        color = 'tab:red'
        ax1_phi.plot(self.x_grid, self.model.phi[des_row, :],
                     color=color, label="phi")
        ax1_phi.set_ylabel(r'$\phi$', color=color)

        color = 'tab:blue'
        ax1_growth = ax1_phi.twinx()
        ax1_growth.plot(self.x_grid, self.model.growth[des_row, :],
                        color=color, label="growth")
        ax1_growth.set_ylabel(r'$\lambda$, -$\nabla \cdot (\phi v_{c})$', color=color)

        if isinstance(self.model.growth_func, PressureConstrainedGrowth):
            full_growth = self.model.growth_func.calc_full_growth(self.model.phi)
            ax1_growth.plot(self.x_grid, full_growth[des_row, :],
                            color='tab:cyan', linestyle='--',
                            label="no_restrict_growth")

        color = 'tab:orange'
        convection = -div2D(self.model.phi * self.model.v_cell,
                            step_size=self.model.GRID,
                            boundary_x="truncate",
                            boundary_y="truncate")
        ax1_growth.plot(self.x_grid, convection[des_row, :], color=color, label="convection")
        ax1_growth.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

        v_ecm = (self.model.v_avg - self.model.phi * self.model.v_cell) / (1 - self.model.phi)
        ax2_velocity.plot(self.x_grid, self.model.v_cell[0, des_row, :], label=r'$v_c$')
        ax2_velocity.plot(self.x_grid, self.model.v_avg[0, des_row, :], label=r'$v_{avg}$')
        ax2_velocity.plot(self.x_grid, v_ecm[0, des_row, :], label=r'$v_e$')

        if self.plot_theory:
            v_theory = self._v_theory()
            ax2_velocity.plot(self.x_grid, v_theory, label=r'$v_{theory}$')

        ax2_velocity.set_ylabel("velocity")
        ax2_velocity.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        ax2_velocity.set_title(f"v_axis, "
                               f"T={self.model.sim_time:.2E}, "
                               f"Δx={self.model.view_moved * self.model.GRID:.2E}")

    def run(self):
        total_frame = len(self.model.sim_time_frame)
        makeAnimation(self.fig, self._draw_single_frame, total_frame,
                      file_name=self.__class__.__name__)
