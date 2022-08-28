from typing import TYPE_CHECKING
import numpy as np

from .base_figure import BaseFigure
from src.tools.plot_tools import neatQuiver, makeAnimation
from src.tools.analyze_tools import Image

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class VelocityField:
    def __init__(self, model):
        self.model: TwoFluidModel = model
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
        ax_cell = self.fig.add_subplot(211)
        ax_avg = self.fig.add_subplot(212)
        edges = Image(self.model.phi).get_edge()
        ax_cell.pcolormesh(self.x_grid, self.y_grid, edges, cmap='gray')
        neatQuiver(self.model.v_cell,
                   grid_size=self.model.GRID,
                   ax=ax_cell,
                   n_points=50)
        ax_cell.set_title(f"v_cell, "
                          f"T={self.model.sim_time:.2E}, "
                          f"Δx={self.model.view_moved * self.model.GRID:.2E}")

        ax_avg.pcolormesh(self.x_grid, self.y_grid, edges, cmap='gray')
        neatQuiver(self.model.v_avg,
                   grid_size=self.model.GRID,
                   ax=ax_avg,
                   n_points=50)
        ax_avg.set_title(f"v_avg, "
                         f"T={self.model.sim_time:.2E}, "
                         f"Δx={self.model.view_moved * self.model.GRID:.2E}")

    def run(self):
        total_frame = len(self.model.sim_time_frame)
        makeAnimation(self.fig, self._draw_single_frame, total_frame,
                      file_name=self.__class__.__name__)
