import matplotlib.pyplot as plt
from src.tools.plot_tools import getFigSize


class BaseFigure:
    def __init__(self, fig_type,
                 n_row_figs: int,
                 n_col_figs: int,
                 **kwargs):
        self.fig_type = fig_type
        self.n_row_figs = n_row_figs
        self.n_col_figs = n_col_figs

        if fig_type == "field":
            self.shape = kwargs["shape"]

        self.fig = self.get_empty_fig()

    def get_empty_fig(self):
        figsize = None
        if self.fig_type == "field":
            figsize = getFigSize(self.shape,
                                 nrows=self.n_row_figs,
                                 ncols=self.n_col_figs)
        elif self.fig_type == "curve":
            figsize = getFigSize((0.6, 1),
                                 nrows=self.n_row_figs,
                                 ncols=self.n_col_figs)

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        return fig
