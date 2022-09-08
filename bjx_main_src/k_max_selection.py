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


def run(job_array_id: int, csv_name_stem: str = None):
    if csv_name_stem is None:
        csv_name_stem = __name__.split('.')[-1]

    param = CsvParamParser(name_stem=csv_name_stem).read_param(job_array_id)

    config = {
        "N_ROW": 75,
        "N_COLUMN": 200,
        "GRID": 1 / 250,
        "T_FINAL": 0.2,
        "T_STEP": 1.5e-7,
        "N_FRAME": 200,
        "PHI_CELL": 0.9,
        "PHI_ECM": 0.1,
        "ZETA_CELL": 2,
        "ZETA_ECM": 6,
        "XI": 0.2,
        "NRG": 0.2,
        "SURF": 5e-5,
        "GROWTH_RATE": 0
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
    p_value = {BoundName.LEFT: -builder.ZETA_CELL * 10,
               BoundName.RIGHT: 0}
    builder.set_p_bound_condition(bound_type=p_type,
                                  bound_value=p_value)
    builder.set_initial_condition(init_shape="half_plane",
                                  x0=95,  # width=15,
                                  perturb_type=PerturbType(param.perturb_type),
                                  noise=param.noise, period=param.period)
    builder.set_free_energy(form="FH")
    builder.set_viscosity(form="step")
    builder.set_growth_function()
    builder.set_pressure_solver(solve_method="traditional")
    builder.set_view_move(engine="tissue_center")
    builder.set_sim_termination(True)

    comment = "[bjx_run]前端生长速度对k的选择性，没有生长，速度恒定为10"
    slurm_info = f"从 {csv_name_stem}.csv 中获取 job_array_id = {job_array_id} 的参数列表{param}"

    model = builder.build_model()

    result_path = "../results"
    save_path = get_save_path(result_path)
    rt_log = RuntimeLogging(filedir=save_path)
    rt_log.START_LOGGING = True
    rt_log.info(comment)
    rt_log.info(slurm_info)

    print("change to ", save_path, flush=True)
    with working_directory(save_path):
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

        plt_script.P0ViProfile(model, plot_theory=False).run()
        plt_script.PhiPressureField(model).run()
        plt_script.VelocityField(model).run()
        plt_script.PressureForceAxis(model, plot_theory=False).run()
        plt_script.VelocityGrowthAxis(model, plot_theory=False).run()
        plt_script.FrontProfile(model)

    print("End plotting")
    rt_log.info("END")
