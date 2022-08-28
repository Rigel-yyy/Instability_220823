import pickle
from typing import TYPE_CHECKING

from src.tools.path_tools import working_directory
import src.plot_src as plt_script

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel

if __name__ == "__main__":
    working_path = r"..\results\220828\018"

    with working_directory(working_path):
        with open('model.pkl', 'rb') as file:
            model: "TwoFluidModel" = pickle.load(file)

        model.load_frame_info()
        print("Load model successfully!")

        # plt_script.PlotP0Profile(model).run()
        plt_script.PhiPressureField(model).run()
        plt_script.VelocityField(model).run()
        plt_script.PressureForceAxis(model, False).run()
        plt_script.VelocityGrowthAxis(model, False).run()
