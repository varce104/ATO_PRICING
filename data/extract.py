# data/extract.py
import numpy as np
import gurobipy as gp
from data.generator import epsilon_ms, delta_ms, lead_times_ms
from data.params import parametros, bill_of_materials, price_set

def extract_params(cfg):
    
    A = bill_of_materials(cfg.size.inst, cfg.size.comp, cfg.size.prod,
                           cfg.bom.min_use, cfg.bom.max_use, cfg.size.seed, cfg.bom.other)
    comp, prod = len(A), len(A[0])

    C, H, pi = parametros(cfg.size.inst, comp, A, cfg.costs.min_cost, cfg.costs.max_cost,
                           cfg.costs.inv_factor, cfg.size.scenarios, cfg.size.seed)
    
    price, a, b, I0 = price_set(cfg.size.inst, cfg.demand.a, cfg.demand.b,
                                 cfg.price.lb_price, cfg.price.ub_price, cfg.price.step_price)
    
    mult = epsilon_ms(cfg.size.inst, cfg.size.stages, cfg.size.scenarios, cfg.size.branching,
                       cfg.size.seed, cfg.demand.lb_epsilon, cfg.demand.ub_epsilon)
    
    add = delta_ms(cfg.size.inst, prod, cfg.size.stages, cfg.size.scenarios, cfg.size.branching,
                    cfg.size.seed, cfg.demand.mu_delta, cfg.demand.std_delta)
    
    L = lead_times_ms(cfg.size.inst, comp, cfg.size.stages, cfg.size.scenarios, cfg.size.branching,
                       cfg.size.seed, cfg.lead_times.lb_L, cfg.lead_times.ub_L, cfg.lead_times.det)

    if I0 is None:
        I0 = [0] * comp
    else:
        I0 = [sum((a - b * I0)*A[i][j] for j in range(prod)) for i in range(comp)]

    return C, H, pi, A, price, mult, add, L, a, b, I0