import pickle
from typing import TYPE_CHECKING

from src.runtime_sim_log import RuntimeLogging
from src.tools.csv_param_tools import CsvParamParser
from src.tools.path_tools import working_directory, get_save_path
from src.tools.pde_tools import BoundType, BoundName
from src.tools.useful_funcs.perturbation import PerturbType
from src.model.builder import ModelBuilder

import src.plot_src as plt_script

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


def run(job_array_id: int, csv_file_name: str = None):
    if csv_file_name is None:
        csv_file_name = __name__.split('.')[-1]

    param = CsvParamParser(name_stem=csv_file_name).read_param(job_array_id)

    config = {
        "N_ROW": 40,
        "N_COLUMN": 120,
        "GRID": 1 / 300,
        "T_FINAL": param.T_FINAL,
        "T_STEP": 1e-7,
        "N_FRAME": param.N_FRAME,
        "PHI_CELL": 0.9,
        "PHI_ECM": 0.1,
        "ZETA_CELL": 1,
        "ZETA_ECM": 2,
        "XI": 0.2,
        "NRG": 0.2,
        "SURF": 6e-5,
        "GROWTH_RATE": 30
    }

    builder = ModelBuilder(config)
    phi_type = {BoundName.UP: BoundType.PERIODIC,
                BoundName.DOWN: BoundType.PERIODIC,
                BoundName.LEFT: BoundType.NEUMANN,
                BoundName.RIGHT: BoundType.NEUMANN}
    phi_value = {BoundName.LEFT: 0,
                 BoundName.RIGHT: 0}
    builder.set_phi_bound_condition(bound_type=phi_type,
                                    bound_value=phi_value)
    mu_type = {BoundName.UP: BoundType.PERIODIC,
               BoundName.DOWN: BoundType.PERIODIC,
               BoundName.LEFT: BoundType.NEUMANN,
               BoundName.RIGHT: BoundType.NEUMANN}
    mu_value = {BoundName.LEFT: 0,
                BoundName.RIGHT: 0}
    builder.set_mu_bound_condition(bound_type=mu_type,
                                   bound_value=mu_value)
    p_type = {BoundName.UP: BoundType.PERIODIC,
              BoundName.DOWN: BoundType.PERIODIC,
              BoundName.LEFT: BoundType.NEUMANN,
              BoundName.RIGHT: BoundType.DIRICHLET}
    p_value = {BoundName.LEFT: 0,
               BoundName.RIGHT: 0}
    builder.set_p_bound_condition(bound_type=p_type,
                                  bound_value=p_value)
    builder.set_initial_condition(init_shape="half_plane",
                                  x0=30,  # width=15,
                                  perturb_type=PerturbType.SINGLE_SINE,
                                  noise=0, period=5)
    builder.set_free_energy(form="FH")
    builder.set_viscosity(form="step")
    builder.set_growth_function(constrain_type="pressure",
                                form="step",
                                threshold=param.threshold,
                                theta_width_ratio=param.ratio)
    builder.set_pressure_solver(solve_method="iterative", max_iter=25)
    builder.set_view_move(engine="off")

    comment = "[bjx_run]在不同的压强threshold限制下，改变压强限制函数看能否逼近理论值"
    slurm_info = f"从 {csv_file_name}.csv 中获取 job_array_id = {job_array_id} 的参数列表{param}"

    model = builder.build_model()

    result_path = "../results"
    save_path = get_save_path(result_path)
    rt_log = RuntimeLogging(filedir=save_path)
    rt_log.START_LOGGING = True
    rt_log.info(comment)
    rt_log.info(slurm_info)

    with working_directory(save_path):
        print("change to ", save_path)
        model.save_model()
        model.set_init()
        model.solvePDE()

    print("End solving PDE")
    rt_log.info("Start plotting simulation results")

    with working_directory(save_path):
        with open('model.pkl', 'rb') as file:
            model: "TwoFluidModel" = pickle.load(file)

        model.load_frame_info()
        print("Load model successfully!")

        plt_script.P0ViProfile(model, plot_theory=True).run()
        plt_script.PhiPressureField(model).run()
        plt_script.VelocityField(model).run()
        plt_script.PressureForceAxis(model, plot_theory=False).run()
        plt_script.VelocityGrowthAxis(model, plot_theory=False).run()

    print("End plotting")
    rt_log.info("END")
