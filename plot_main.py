import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from src.tools.path_tools import working_directory
import src.plot_src as plt_script

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel

if __name__ == "__main__":
    date_str = '220902'
    order_str = '009'
    working_path = Path('..').joinpath('results').\
        joinpath(date_str).\
        joinpath(order_str)

    with working_directory(working_path):
        with open('model.pkl', 'rb') as file:
            model: "TwoFluidModel" = pickle.load(file)

        model.load_frame_info()
        print("Load model successfully!")

        plt_script.P0ViProfile(model, plot_theory=True).run()
        plt_script.PhiPressureField(model).run()
        plt_script.VelocityField(model).run()
        plt_script.PressureForceAxis(model, plot_theory=False).run()
        plt_script.VelocityGrowthAxis(model, plot_theory=False).run()
