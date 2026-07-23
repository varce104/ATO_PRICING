import copy
import math
from data.config import (ProblemSize, BomConfig, CostConfig, PriceConfig,
                          DemandConfig, LeadTimeConfig, RunConfig, ExperimentConfig)
from pipeline.experiments import run_parameter, apply_comp, apply_prod, apply_branching, apply_phi_mode

# --- Instancia base (idéntica a la de main.py) ---
comp, prod, stages = 5, 4, 8
branching = [2]*7
scenarios = math.prod(branching)

size = ProblemSize("None", comp, prod, stages, scenarios, branching, seed=5, time_limit=900)
bom = BomConfig(min_use=3, max_use=4, other=False)
costs = CostConfig(min_cost=5, max_cost=25, inv_factor=0.2, I0=60)
price = PriceConfig(lb_price=15, ub_price=60, step_price=5)
demand = DemandConfig(a=100, b=1.6, lb_epsilon=0.7, ub_epsilon=1.3, mu_delta=0, std_delta=2)
lead_times = LeadTimeConfig(lb_L=1, ub_L=2, det=False)

run = RunConfig(Model="MS_linear", show_fulfillment=False)

base_cfg = ExperimentConfig(size, bom, costs, price, demand, lead_times, run, iter=3)

# --- Eje 1: número de componentes ---
run_parameter(base_cfg, "comp", [4, 5, 6, 7, 8, 9, 10], apply_comp)

# --- Eje 2: número de productos ---
run_parameter(base_cfg, "prod", [3, 4, 5, 6, 7, 8, 9], apply_prod)

# --- Eje 3: branching (definir tus propias formas de árbol aquí) ---
branching_options = [
    [2,2,2,2,2,2,2],    
    [5,5,5,2,1,1,1],
    [10,5,2,2,1,1,1],
    [20,5,2,1,1,1,1],
    [50,2,2,1,1,1,1],
    [125,2,1,1,1,1,1],
]
run_parameter(base_cfg, "branching", branching_options, apply_branching)

# --- Eje 4: modo de phi (para modelos afines) ---
affine_cfg = copy.deepcopy(base_cfg)
affine_cfg.run = RunConfig(Model="Affine_eval",)
run_parameter(affine_cfg, "phi_mode", ["eps", "eps_delta", "eps_lt", "eps_delta_lt"], apply_phi_mode)