from typing import TYPE_CHECKING
import numpy as np

from .base_figure import BaseFigure
from src.tools.plot_tools import makeAnimation

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class PhiPressureField:
    def __init__(self, model):
        self.model: "TwoFluidModel" = model
        self.fig = BaseFigure(fig_type="field",
                              n_row_figs=2,
                              n_col_figs=1,
                              shape=(model.N_ROW, model.N_COLUMN)
                              ).fig
        self.x_grid = np.arange(model.N_COLUMN) * model.GRID
        self.y_grid = np.arange(model.N_ROW) * model.GRID

    def _draw_single_frame(self, n_frame):
        self.model.load_frame(n_frame)
        self.fig.clear()

        ax_phi = self.fig.add_subplot(211)
        ax_pressure = self.fig.add_subplot(212)

        image_phi = ax_phi.pcolormesh(self.x_grid, self.y_grid, self.model.phi,
                                      cmap='plasma', shading='auto')
        ax_phi.set_title(f"phi, "
                         f"T={self.model.sim_time:.2E}, "
                         f"Δx={self.model.view_moved * self.model.GRID:.2E}")

        self.fig.colorbar(mappable=image_phi, ax=ax_phi)

        image_pressure = ax_pressure.pcolormesh(self.x_grid, self.y_grid,
                                                self.model.pressure,
                                                cmap='plasma', shading='auto')
        ax_pressure.set_title(f"pressure, "
                              f"T={self.model.sim_time:.2E}, "
                              f"Δx={self.model.view_moved * self.model.GRID:.2E}")
        self.fig.colorbar(mappable=image_pressure, ax=ax_pressure)

    def run(self):
        total_frame = len(self.model.sim_time_frame)
        makeAnimation(self.fig, self._draw_single_frame, total_frame,
                      file_name=self.__class__.__name__)
