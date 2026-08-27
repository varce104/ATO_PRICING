from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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
    oos_branching: Optional[List[int]] = None  # Params OOS
    oos_scenarios: Optional[int] = None

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
    a: float; b: float;                     # Linear demand function 
    lb_epsilon: float; ub_epsilon: float    # Mult stochastic component
    mu_delta: float; std_delta: float       # Add sto0chastic component

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

@dataclass
class OutputConfig:
    show_var: bool = False
    lambda_app: bool = False
    show_heatmap: bool = False
    show_boxplot: bool = False
    show_candlestick: bool = False
    show_kpis: bool = False
    vss_calc: bool = False
    evpi_calc: bool = False
    vss_ts_calc: bool = False

@dataclass
class ExperimentConfig:
    size: ProblemSize; bom: BomConfig; costs: CostConfig
    price: PriceConfig; demand: DemandConfig; lead_times: LeadTimeConfig
    run: RunConfig; Output: OutputConfig; iter: int = 1