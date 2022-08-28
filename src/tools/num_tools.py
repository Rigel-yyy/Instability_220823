import numpy as np
import math
from numba import jit
from scipy.ndimage import convolve, convolve1d
from .type_tools import ScalarField2D, VectorField2D


def laplacian2D(mat: ScalarField2D,
                step_size: float = 1.,
                boundary_x: str = 'wrap',
                boundary_y: str = 'wrap') -> ScalarField2D:
    """
    wrap option means periodic boundary condition
    nearest option means open boundary condition
    """
    if boundary_x == 'truncate' or boundary_y == 'truncate':
        del_mat = grad2D(mat, step_size, boundary_x, boundary_y)
        return div2D(del_mat, step_size, boundary_x, boundary_y)
    else:
        if boundary_x == boundary_y:
            stencil = 1 / step_size ** 2 * np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=np.float64)
            return convolve(mat.astype(float), stencil, mode=boundary_x)
        else:
            stencil = np.array([1, -2, 1], dtype=np.float64) / (step_size ** 2)
            d2f_dx2 = convolve1d(mat.astype(float), stencil, axis=1, mode=boundary_x)
            d2f_dy2 = convolve1d(mat.astype(float), stencil, axis=0, mode=boundary_y)
            return d2f_dx2 + d2f_dy2


def grad2D(mat: ScalarField2D,
           step_size: float = 1.,
           boundary_x: str = 'wrap',
           boundary_y: str = 'wrap') -> VectorField2D:
    """
    gradient of scalar field

    Parameters
    ----------
    mat : ScalarField2D
    step_size : float, default is 1.
    boundary_x : str, default is 'wrap'.
    boundary_y : str, default is 'wrap'.

    Returns
    -------
    VectorField2D
    """

    stencil = np.array([1, 0, -1], dtype=np.float64) / (2 * step_size)
    if boundary_x != "truncate":
        df_dx = convolve1d(mat.astype(float), stencil, axis=1, mode=boundary_x)
    else:
        df_dx = np.gradient(mat, axis=1, edge_order=2) / step_size

    if boundary_y != "truncate":
        df_dy = convolve1d(mat.astype(float), stencil, axis=0, mode=boundary_y)
    else:
        df_dy = np.gradient(mat, axis=0, edge_order=2) / step_size
    return np.array([df_dx, df_dy])


def div2D(vmat: VectorField2D,
          step_size: float = 1.,
          boundary_x: str = 'wrap',
          boundary_y: str = 'wrap') -> ScalarField2D:
    """
    divergence of vector field

    Parameters
    ----------
    vmat : VectorField2D
    step_size : float, default is 1
    boundary_x : str, default is 'wrap'
        boundary condition used when calculating d/dx
    boundary_y : str, default is 'wrap'
        boundary condition used when calculating d/dy

    Returns
    -------
    ScalarField2D
    """

    fx, fy = vmat
    stencil = np.array([1, 0, -1], dtype=np.float64) / (2 * step_size)
    if boundary_x != "truncate":
        dfx_dx = convolve1d(fx.astype(float), stencil, axis=1, mode=boundary_x)
    else:
        dfx_dx = np.gradient(fx, axis=1, edge_order=2) / step_size

    if boundary_y != "truncate":
        dfy_dy = convolve1d(fy.astype(float), stencil, axis=0, mode=boundary_y)
    else:
        dfy_dy = np.gradient(fy, axis=0, edge_order=2) / step_size

    return dfx_dx + dfy_dy


def inner2D(vmat1: VectorField2D, vmat2: VectorField2D) -> ScalarField2D:
    """
    inner product of two vector field
    """

    return np.sum(vmat1 * vmat2, axis=0)


@jit
def has_nan(mat: np.ndarray):
    for idx, item in enumerate(mat.ravel()):
        if math.isnan(item):
            return idx
    return None
