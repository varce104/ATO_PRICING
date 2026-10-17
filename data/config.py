from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

@dataclass
class ProblemSize:
    inst: Optional[str];        # Special instance
    comp: int;                  # Components
    prod: int;                  # Products
    stages: int                 # Periods/stages
    scenarios: int;             # Scenarios
    branching: List[int];       # Scenario tree branching
    seed: int;                  # Seed
    time_limit: int             # Runtime limit
    scenario_groups: Optional[Dict[int, List[List[int]]]] = None

@dataclass
class BomConfig:
    min_use: int;               # Min use of component in order to assemble a product
    max_use: int;               # Max use of component in order to assemble a product
    other: bool                 # Specific BoM from params.py

@dataclass
class CostConfig:
    min_cost: int;              # Min procurement cost
    max_cost: int;              # Max procurement cost
    inv_factor: float;          # Ratio of Procurement cost - Inventory cost
    I0: float               # Initial inventory intended to satisfy expected demandat given price

@dataclass
class PriceConfig:
    lb_price: int;              # Lower bound of price
    ub_price: int;              # Upper bound of price
    step_price: int             # price interval for set

@dataclass
class DemandConfig:
    a: any; b: any;                     # Linear demand function 
    lb_epsilon: float; ub_epsilon: float    # Mult stochastic component
    mu_delta: float; std_delta: float       # Add sto0chastic component
    a_range: Optional[Tuple[float, float]] = None   # si se define: a_j ~ U[range], ignora `a`
    b_range: Optional[Tuple[float, float]] = None

@dataclass
class LeadTimeConfig:
    lb_L: int; ub_L: int;       # Lead time distribution  
    det: bool                   # If lead time is deterministic or stochastic

@dataclass
class RunConfig:
    Model: str
    W_cts: bool = False
    features: bool = False
    phi_mode: str = "eps_delta"   # "eps" | "eps_delta" | "eps_lt" | "eps_delta_lt"
    local_search: bool = False # True if local search applied to MS_linear, with AF_EVAL initial solution (or other)
    oos_eval: bool = False    # True if out-of-sample evaluation of MS_linear with AF_EVAL solution
    max_iter: int = 1
    # Rolling horizon parameters
    rh_horizon: int = 3
    rh_fix_x: bool = True       # True: evalúa la política RH completa; False: solo fija w (comparable con AF_EVAL)
    rh_mip_gap: float = 1e-3
    rh_sub_time: int = 60       # límite por ventana
    rh_resample: bool = False

@dataclass
class OutputConfig:
    show_sol: bool = False
    show_heatmap: bool = False
    show_boxplot: bool = False
    show_candlestick: bool = False
    show_kpis: bool = False
    vss_calc: bool = False
    evpi_calc: bool = False
    vss_ts_calc: bool = False

@dataclass
class ExperimentConfig:
    size: ProblemSize; 
    bom: BomConfig; 
    costs: CostConfig
    price: PriceConfig; 
    demand: DemandConfig; 
    lead_times: LeadTimeConfig
    run: RunConfig; 
    Output: OutputConfig; 
    iter: int = 1

    