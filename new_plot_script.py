# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:23:55 2022

@author: Rigel_yyy
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

from src.tools.num_tools import grad2D, laplacian2D, div2D
from src.tools.plot_tools import neatQuiver, makeAnimation, getFigSize
from src.tools.path_tools import working_directory
from src.tools.analyze_tools import Image
from src.model.two_fluid_model import TwoFluidModel

if __name__ == "__main__":

    working_path = r"..\results\220825\004"

    with working_directory(working_path):
        with open('model.pkl', 'rb') as file:
            model: TwoFluidModel = pickle.load(file)

        model.load_frame_info()
        print("Load model successfully!")

    x_grid = np.arange(model.N_COLUMN) * model.GRID
    y_grid = np.arange(model.N_ROW) * model.GRID

    figsize = getFigSize((model.N_ROW, model.N_COLUMN), nrows=2)  # TODO
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    



    def theory_uniform_case():
        phi_binary_img = Image(model.phi).get_binary_image(model.PHI_CELL-0.2, 1)
        cell_x_width = phi_binary_img.get_x_width()
        cell_y_width = model.N_ROW
        growth_eff = np.sum(model.growth) / (cell_x_width * cell_y_width)
        interface = model.GRID * cell_x_width

        v_profile = np.where(x_grid < interface,
                             growth_eff * x_grid,
                             growth_eff * interface)
        v_ecm = interface * growth_eff
        p_profile = np.where(x_grid < interface,
                             -0.5 * model.ZETA_CELL * growth_eff * (x_grid ** 2),
                             -0.5 * model.ZETA_CELL * growth_eff * (interface ** 2)
                             - v_ecm * model.ZETA_ECM * (x_grid - interface))
        p_shift = np.min(p_profile)
        return v_profile, p_profile - p_shift











    with working_directory(working_path):
        total_frame = len(model.sim_time_frame)
        makeAnimation(fig, phi_pressure_field, total_frame)
        makeAnimation(fig, x_axis_distribution, total_frame)
        makeAnimation(fig, pressure_axis, total_frame)
        makeAnimation(fig, v_vector_field, total_frame)
        # plot_p0_profile()
