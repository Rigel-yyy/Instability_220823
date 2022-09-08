from typing import Iterable, TYPE_CHECKING

import numpy as np
from scipy import interpolate

from .base_figure import BaseFigure
from src.tools.analyze_tools import Image
from src.tools.day_tools import getTimeStamp


if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class FrontProfile:
    def __init__(self, model, x_cpt_ratio=0.75, plot_exp_growth=False):
        self.fig = BaseFigure(fig_type="curve",
                              n_col_figs=1,
                              n_row_figs=1).fig
        self.model: "TwoFluidModel" = model
        self.x_checkpoint = self.model.N_COLUMN * self.model.GRID * x_cpt_ratio
        self.plot_exp_growth = plot_exp_growth

    def run(self, frame_range: Iterable = None):
        if frame_range is None:
            max_frame = len(self.model.sim_time_frame)
            frame_range = range(max_frame)

        x_front_arr = []
        t_arr = []
        grow_area_arr = []
        interface_area_arr = []
        unit_area = self.model.GRID ** 2

        for n_frame in frame_range:
            self.model.load_frame(n_frame)
            phi_img = Image(self.model.phi)
            front_idx = phi_img.get_front_most(direction="right")
            n_growth_lattices = np.sum(phi_img.get_binary_image(self.model.PHI_CELL - 0.01, 1).data)
            n_interface_lattices = np.sum(phi_img.get_binary_image(0.5, 1).data) - n_growth_lattices
            if front_idx is not None:
                t_arr.append(self.model.sim_time)
                x_front_arr.append(front_idx * self.model.GRID)
                grow_area_arr.append((n_growth_lattices + self.model.view_moved) * unit_area)
                interface_area_arr.append(n_interface_lattices * unit_area)

        x_front_arr = np.array(x_front_arr)
        t_arr = np.array(t_arr)
        grow_area_arr = np.array(grow_area_arr)
        interface_area_arr = np.array(interface_area_arr)
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

        color = 'tab:purple'
        Ly = self.model.N_ROW * self.model.GRID
        ax_front.plot(t_arr, (grow_area_arr + interface_area_arr) / Ly,
                      label="sim_area / Ly", color=color)

        if self.plot_exp_growth:
            theory_area_arr = grow_area_arr[0] * np.exp(self.model.GROWTH_RATE * t_arr)
            ax_front.plot(t_arr, (theory_area_arr + interface_area_arr) / Ly,
                          linestyle='-.', color='sienna', label="theory_area / Ly")

        ax_front.legend()
        ax_front.set_title("x front profile")
        ax_front.set_xlabel("t")
        ax_front.set_ylabel("x front")

        self.fig.savefig(getTimeStamp() + "x_front_profile" + ".pdf")
