from typing import TYPE_CHECKING
import numpy as np

from .base_figure import BaseFigure
from src.tools.analyze_tools import Image
from src.tools.day_tools import getTimeStamp

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class PlotP0Profile:
    def __init__(self, model):
        self.fig = BaseFigure(fig_type="curve",
                              n_row_figs=1,
                              n_col_figs=1).fig
        self.model: "TwoFluidModel" = model

    def run(self, frame_range=None):
        if frame_range is None:
            max_frame = len(self.model.sim_time_frame)
            frame_range = range(max_frame)

        xi_arr = np.zeros(len(frame_range))
        p0_sim_arr = np.zeros(len(frame_range))

        for n_frame in frame_range:
            self.model.load_frame(n_frame)
            phi_binary_img = Image(self.model.phi).get_binary_image(self.model.PHI_CELL - 0.2, 1)
            xi = phi_binary_img.get_x_width() * self.model.GRID
            p0 = np.mean(self.model.pressure[:, 3])
            xi_arr[n_frame] = xi
            p0_sim_arr[n_frame] = p0

        ax = self.fig.add_subplot(111)
        ax.scatter(xi_arr, p0_sim_arr, label="p0_sim", s=2)
        ax.axvline(self.model.N_COLUMN * self.model.GRID /
                   (2 - self.model.ZETA_CELL / self.model.ZETA_ECM),
                   color='tab:red')
        ax.legend()
        ax.set_title("p0 profile")
        ax.set_xlabel("$x_I$")
        ax.set_ylabel("$p_0$")
        self.fig.savefig(getTimeStamp() + "p0_profile" + ".pdf")
