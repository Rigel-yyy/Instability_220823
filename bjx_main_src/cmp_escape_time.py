import pickle
import copy
from typing import TYPE_CHECKING

from src.runtime_sim_log import RuntimeLogging
from src.tools.csv_param_tools import CsvParamParser
from src.tools.path_tools import working_directory, get_save_path
from src.tools.pde_tools import BoundType, BoundName
from src.tools.useful_funcs.perturbation import PerturbType
from src.model.builder import ModelBuilder
from src.model.base_model import BaseModel

import src.plot_src as plt_script

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel


def run(job_array_id: int, csv_name_stem: str = None):
    if csv_name_stem is None:
        csv_name_stem = __name__.split('.')[-1]

    param = CsvParamParser(name_stem=csv_name_stem).read_param(job_array_id)

    run_config = {
        "N_ROW": 75,
        "N_COLUMN": 180,
        "GRID": 1 / 250,
        "T_FINAL": 0.1,
        "T_STEP": 1e-7,
        "N_FRAME": 200,
        "PHI_CELL": 0.9,
        "PHI_ECM": 0.1,
        "ZETA_CELL": 2,
        "ZETA_ECM": 6,
        "XI": 0.2,
        "NRG": 0.2,
        "SURF": 5e-5,
        "GROWTH_RATE": 40
    }

    warm_up_config = copy.deepcopy(run_config)
    warm_up_config["T_FINAL"] = 200 * run_config["T_STEP"]
    warm_up_config["N_FRAME"] = 5

    warm_up_builder = ModelBuilder(warm_up_config)
    phi_type = {BoundName.UP: BoundType.PERIODIC,
                BoundName.DOWN: BoundType.PERIODIC,
                BoundName.LEFT: BoundType.NEUMANN,
                BoundName.RIGHT: BoundType.NEUMANN}
    phi_value = {BoundName.LEFT: 0,
                 BoundName.RIGHT: 0}
    warm_up_builder.set_phi_bound_condition(bound_type=phi_type,
                                            bound_value=phi_value)
    mu_type = {BoundName.UP: BoundType.PERIODIC,
               BoundName.DOWN: BoundType.PERIODIC,
               BoundName.LEFT: BoundType.NEUMANN,
               BoundName.RIGHT: BoundType.NEUMANN}
    mu_value = {BoundName.LEFT: 0,
                BoundName.RIGHT: 0}
    warm_up_builder.set_mu_bound_condition(bound_type=mu_type,
                                           bound_value=mu_value)
    p_type = {BoundName.UP: BoundType.PERIODIC,
              BoundName.DOWN: BoundType.PERIODIC,
              BoundName.LEFT: BoundType.NEUMANN,
              BoundName.RIGHT: BoundType.DIRICHLET}
    p_value = {BoundName.LEFT: 0,
               BoundName.RIGHT: 0}
    warm_up_builder.set_p_bound_condition(bound_type=p_type,
                                          bound_value=p_value)
    warm_up_builder.set_initial_condition(init_shape="half_plane",
                                          x0=30,
                                          perturb_type=PerturbType.MULTI_SINE,
                                          noise=param.noise, max_period=8)
    warm_up_builder.set_free_energy(form="FH")
    warm_up_builder.set_viscosity(form="step")
    warm_up_builder.set_growth_function(phi_threshold=1)
    warm_up_builder.set_pressure_solver(solve_method="traditional")
    warm_up_builder.set_view_move(engine="off")
    warm_up_builder.set_sim_termination(True)

    warm_up_comment = "[bjx_warm_up]运行warm up model保证正式运行时压强迭代能收敛"

    warm_up_model = warm_up_builder.build_model()

    result_path = "../results"
    save_path = get_save_path(result_path)
    rt_log = RuntimeLogging(filedir=save_path)
    rt_log.START_LOGGING = True
    rt_log.info(warm_up_comment)

    print("change to ", save_path, flush=True)
    warm_up_path = save_path.joinpath('warm_up')
    warm_up_path.mkdir()
    with working_directory(warm_up_path):
        warm_up_model.save_model()
        warm_up_model.set_init()
        warm_up_model.solvePDE()
    rt_log.info("Warm up finished successfully")

    BaseModel.set_config(run_config)
    run_builder = copy.deepcopy(warm_up_builder)
    run_builder.set_growth_function(constrain_type="pressure",
                                    form="step",
                                    p_threshold=param.p_threshold,
                                    theta_width_ratio=0.025,
                                    phi_threshold=1)
    run_builder.set_pressure_solver(solve_method="iterative", max_iter=25)
    run_model = run_builder.build_model()

    run_comment = "[bjx_run]比较在不同压强限制下有不稳定性和没有不稳定性的逃逸时间"
    rt_log.info(run_comment)

    with working_directory(save_path):
        run_model.save_model()
        run_model.copy_status(warm_up_model)
        run_model.solvePDE()

    print("End solving PDE")
    rt_log.info("Start plotting simulation results")

    with working_directory(save_path):
        with open('model.pkl', 'rb') as file:
            model: "TwoFluidModel" = pickle.load(file)

        model.load_frame_info()
        print("Load model successfully!")

        if param.plot_theory:
            plt_script.P0ViProfile(model, plot_theory=True).run()

        plt_script.PhiPressureField(model).run()
        plt_script.VelocityField(model).run()
        plt_script.PressureForceAxis(model, plot_theory=param.plot_theory).run()
        plt_script.VelocityGrowthAxis(model, plot_theory=param.plot_theory).run()
        plt_script.FrontProfile(model).run()

    print("End plotting")
    rt_log.info("END")
