from typing import Iterable, TYPE_CHECKING

import numpy as np
from scipy import interpolate

from .base_figure import BaseFigure
from src.tools.analyze_tools import Image
from src.tools.day_tools import getTimeStamp


if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class FrontProfile:
    def __init__(self, model, x_cpt_ratio=0.75):
        self.fig = BaseFigure(fig_type="curve",
                              n_col_figs=1,
                              n_row_figs=1).fig
        self.model: "TwoFluidModel" = model
        self.x_checkpoint = self.model.N_COLUMN * self.model.GRID * x_cpt_ratio

    def run(self, frame_range: Iterable = None):
        if frame_range is None:
            max_frame = len(self.model.sim_time_frame)
            frame_range = range(max_frame)

        x_front_arr = []
        t_arr = []

        for n_frame in frame_range:
            self.model.load_frame(n_frame)
            front_idx = Image(self.model.phi).get_front_most(direction="right")
            if front_idx is not None:
                t_arr.append(self.model.sim_time)
                x_front_arr.append(front_idx * self.model.GRID)

        x_front_arr = np.array(x_front_arr)
        t_arr = np.array(t_arr)
        t_checkpoint = interpolate.interp1d(x_front_arr,
                                            t_arr,
                                            fill_value="extrapolate")(self.x_checkpoint)

        ax_front = self.fig.add_subplot(111)
        ax_front.plot(t_arr, x_front_arr, label="x_front")
        ax_front.scatter([t_checkpoint], [self.x_checkpoint], color='tab:red')
        ax_front.annotate(f"({t_checkpoint:.4e},{self.x_checkpoint:.4e})",
                          xy=(t_checkpoint, self.x_checkpoint),
                          color="tab:red",
                          xytext=(0.8*t_checkpoint, self.x_checkpoint*1.2),
                          arrowprops=dict(facecolor='orange', shrink=0.05),
                          ha="center")
        ax_front.axhline(self.x_checkpoint, linestyle='--', color='tab:orange')
        ax_front.set_title("x front profile")
        ax_front.set_xlabel("t")
        ax_front.set_ylabel("x front")

        self.fig.savefig(getTimeStamp() + "x_front_profile" + ".pdf")
