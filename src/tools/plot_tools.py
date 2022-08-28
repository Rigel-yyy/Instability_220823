from .type_tools import ScalarField2D, VectorField2D
from .day_tools import getTimeStamp
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple, Callable
import numpy as np


def getFigSize(shape: Tuple[int],
               nrows: int = 1,
               ncols: int = 1):
    inner_size_x = 6 * ncols
    inner_size_y = 6 * nrows * shape[0] / shape[1]
    frame_size_x = 1 * ncols + 0.2 * (ncols - 1)
    frame_size_y = 0.6 * nrows + 0.2 * (nrows - 1)
    return inner_size_x + frame_size_x, inner_size_y + frame_size_y


def scalarFieldRel2D(field1: ScalarField2D,
                     field2: ScalarField2D,
                     n_points: int = None,
                     cutoff: float = 1):
    """
    get field correlation between field1 and field2

    --------
    field1 : ScalarField2D, value as x
    field2 : ScalarField2D, value as y
    n_points : int, n_points bins in return
    cutoff : float, ignore field1 > np.max(field1)*cutoff
    """

    if field1.shape != field2.shape:
        raise ValueError("Inconsistent matrix size between field1 and field2!")

    if n_points is None:
        n_points = np.max(field1.shape) // 2

    # sort according to field1
    idx = np.argsort(field1.flatten())
    field1 = field1.flatten()[idx]
    field2 = field2.flatten()[idx]

    # set bin edges
    edge = np.linspace(field1[0], cutoff * field1[-1], num=n_points + 1)
    edge_idx = np.searchsorted(field1, edge)

    x_result = []
    y_result = []
    for i in range(n_points):
        left = edge_idx[i]
        right = edge_idx[i + 1]
        if left == right:
            break
        else:
            x_result.append(np.mean(field1[left: right]))
            y_result.append(np.mean(field2[left: right]))

    return np.array(x_result), np.array(y_result)


def neatQuiver(vmat: VectorField2D,
               grid_size: float,
               ax: Axes,
               n_points: int = 30):
    """
    plot neat quiver plot with colormap

    ----------
    vmat : VectorField2D to be plotted
    grid_size : float
    ax : matplotlib.axes.Axes
    n_points : int, optional
        number of vector in one axis. The default is 30.
    """

    _, nrows, ncols = vmat.shape
    if ncols > n_points:
        x_idx = np.round(np.linspace(0, ncols - 1, n_points)).astype(int)
    else:
        x_idx = np.arange(ncols)

    if nrows > n_points:
        y_idx = np.floor(np.linspace(0, nrows - 1, n_points)).astype(int)
    else:
        y_idx = np.arange(nrows)

    x_mesh, y_mesh = np.meshgrid(x_idx * grid_size,
                                 y_idx * grid_size)
    xx_idx, yy_idx = np.meshgrid(x_idx, y_idx)
    v_x = vmat[0, yy_idx, xx_idx]
    v_y = vmat[1, yy_idx, xx_idx]

    color = np.hypot(v_x, v_y)
    norm = mpl.colors.Normalize()
    norm.autoscale(color)
    sm = mpl.cm.ScalarMappable(norm=norm)
    sm.set_array([])

    scale = np.max(color) * np.max([x_idx.size, y_idx.size]) / 3
    ax.quiver(x_mesh, y_mesh,
              v_x, v_y, color,
              scale_units='height', scale=scale)
    ax.get_figure().colorbar(sm, ax=ax)


def makeAnimation(fig: Figure,
                  func: Callable[[int], None],
                  n_frame: int,
                  file_name: str):
    """
    generate animation according to plot function func

    Parameters
    ----------
    fig : mpl.figure.Figure
        figure to plot animation on
    func : Callable[[int],None]
        function with frame num as the only parameter
    n_frame : int
        total frame number to plot
    file_name : str
        file name of the mp4 file
    """

    ani = animation.FuncAnimation(fig, func,
                                  frames=range(n_frame),
                                  interval=200)
    ani.save(getTimeStamp() + file_name + ".mp4",
             extra_args=['-vcodec', 'libx264'],
             dpi=300)
