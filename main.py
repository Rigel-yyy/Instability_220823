import pickle
from typing import TYPE_CHECKING

from src.runtime_sim_log import RuntimeLogging
from src.tools.path_tools import working_directory, get_save_path
from src.tools.pde_tools import BoundType, BoundName
from src.tools.useful_funcs.perturbation import PerturbType
from src.model.builder import ModelBuilder

if TYPE_CHECKING:
    from src.model.two_fluid_model import TwoFluidModel

if __name__ == "__main__":
    # init_path = r"..\results\220828\015"
    #
    # with working_directory(init_path):
    #     with open('model.pkl', 'rb') as file:
    #         init_model: "TwoFluidModel" = pickle.load(file)
    #     init_model.load_frame_info()
    #     init_model.load_frame(len(init_model.sim_time_frame)-1)

    config = {
        "N_ROW": 80,
        "N_COLUMN": 120,
        "GRID": 1 / 250,
        "T_FINAL": 1e-4,
        "T_STEP": 1e-7,
        "N_FRAME": 10,
        "PHI_CELL": 0.9,
        "PHI_ECM": 0.1,
        "ZETA_CELL": 2,
        "ZETA_ECM": 5,
        "XI": 0.2,
        "NRG": 0.2,
        "SURF": 5e-5,
        "GROWTH_RATE": 40
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
                                  perturb_type=PerturbType.MULTI_SINE,
                                  noise=3, max_period=8)
    builder.set_free_energy(form="FH")
    builder.set_viscosity(form="step")
    builder.set_growth_function(phi_threshold=1)
    # builder.set_growth_function(constrain_type="pressure",
    #                             form="step",
    #                             p_threshold=1.4,
    #                             theta_width_ratio=0.025,
    #                             phi_threshold=0.65)
    builder.set_pressure_solver(solve_method="traditional")
    # builder.set_pressure_solver(solve_method="iterative", max_iter=25)
    builder.set_view_move(engine="off")

    comment = "[init_prepare]为GROWTH_RATE=30的有微扰受限生长准备初态"

    model = builder.build_model()

    result_path = "../results"
    save_path = get_save_path(result_path)
    rt_log = RuntimeLogging(filedir=save_path)
    rt_log.START_LOGGING = True
    rt_log.info(comment)

    with working_directory(save_path):
        print("change to ", save_path)
        model.save_model()
        model.set_init()
        # model.copy_status(init_model)
        model.solvePDE()

    print("end")
