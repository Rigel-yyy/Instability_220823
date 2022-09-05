from typing import Iterable, TYPE_CHECKING
import numpy as np

from .base_figure import BaseFigure
from src.tools.analyze_tools import Image
from src.tools.day_tools import getTimeStamp
from src.model.active_growth import (FullGrowth,
                                     PressureConstrainedGrowth)

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


class P0ViProfile:
    def __init__(self, model, plot_theory=False):
        self.fig = BaseFigure(fig_type="curve",
                              n_row_figs=2,
                              n_col_figs=1).fig
        self.plot_theory = plot_theory
        self.model: "TwoFluidModel" = model

    def plot_xI_star_theory(self, ax):
        # x_I^* 是在生长没有压强限制时计算出的
        xI_star = self.model.N_COLUMN * self.model.GRID / \
                  (2 - self.model.ZETA_CELL / self.model.ZETA_ECM)
        ax.axvline(xI_star, linestyle='--', color='tab:orange',
                   label="$x_I = x_I^*$")

    def plot_vI_theory(self, ax, xI):
        if isinstance(self.model.growth_func, FullGrowth):
            vI = self.model.GROWTH_RATE * xI

        elif isinstance(self.model.growth_func, PressureConstrainedGrowth):
            ax.plot(xI, self.model.GROWTH_RATE * xI,
                    color='tab:orange', linestyle='--',
                    label="$v_I$_no_res")
            L = self.model.N_COLUMN * self.model.GRID
            alpha = self.model.ZETA_CELL / self.model.ZETA_ECM
            criteria = 1 / (2 - alpha)
            pc_threshold = self.model.growth_func.constrain_func.threshold
            pc_norm = 2 * pc_threshold / (self.model.GROWTH_RATE * self.model.ZETA_ECM * L**2)

            if pc_norm >= criteria:
                vI = self.model.GROWTH_RATE * xI
            else:
                xI_norm = xI / L
                xL_norm = criteria - np.sqrt(criteria ** 2 - criteria * pc_norm)
                xR_norm = criteria + np.sqrt(criteria ** 2 - criteria * pc_norm)
                cons_mask = (xL_norm < xI_norm) & (xR_norm > xI_norm)

                tmp = (1 - xI_norm) / alpha
                vI_norm = np.where(cons_mask,
                                   -tmp + np.sqrt(tmp**2 + pc_norm / alpha),
                                   xI_norm)
                vI = L * self.model.GROWTH_RATE * vI_norm
        else:
            return

        ax.plot(xI, vI, color='tab:red', label="$v_I$_theory")

    def run(self, frame_range: Iterable = None):
        if frame_range is None:
            max_frame = len(self.model.sim_time_frame)
            frame_range = range(max_frame)

        xI_arr = np.zeros(len(frame_range))
        p0_sim_arr = np.zeros(len(frame_range))
        vI_sim_arr = np.zeros(len(frame_range))

        for n_frame in frame_range:
            self.model.load_frame(n_frame)
            # 因为只需要算速度，用能生长的区域确定边界更为合理
            phi_binary_img = Image(self.model.phi).\
                get_binary_image(self.model.PHI_CELL-0.005, 1)

            xI = phi_binary_img.get_x_width() * self.model.GRID
            p0 = np.mean(self.model.pressure[:, :3])
            vI = np.mean(self.model.v_avg[0, :, -3:])
            xI_arr[n_frame] = xI
            p0_sim_arr[n_frame] = p0
            vI_sim_arr[n_frame] = vI

        ax1_p0 = self.fig.add_subplot(211)
        ax2_vI = self.fig.add_subplot(212)

        ax1_p0.scatter(xI_arr, p0_sim_arr, label="$p_0$_sim", s=2)
        if self.plot_theory:
            self.plot_xI_star_theory(ax1_p0)

        ax1_p0.legend()
        ax1_p0.set_title("$p_0$ profile")
        ax1_p0.set_xlabel("$x_I$")
        ax1_p0.set_ylabel("$p_0$")

        ax2_vI.scatter(xI_arr, vI_sim_arr, label="$v_I$_sim", s=2)
        if self.plot_theory:
            self.plot_vI_theory(ax2_vI, xI_arr)

        ax2_vI.legend()
        ax2_vI.set_title("$v_I$ profile")
        ax2_vI.set_xlabel("$x_I$")
        ax2_vI.set_ylabel("$v_I$")

        self.fig.savefig(getTimeStamp() + "p0_profile" + ".pdf")
