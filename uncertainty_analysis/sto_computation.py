from data.generator import epsilon_ms, delta_ms, lead_times_ms
from data.params import parametros, bill_of_materials, price_set
from uncertainty_analysis.vss import VSS
from uncertainty_analysis.evpi import EVPI
from uncertainty_analysis.vss_ts import VSS_2S

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def extract_params(size, bom, costs, price_param, demand, lead_times):

    inst, comp, prod, stages, scenarios, branching, seed, _ = size
    min_use, max_use, other = bom
    min_cost, max_cost, inv_factor, I0 = costs
    lb_price, ub_price, step_price = price_param
    a, b, lb_epsilon, ub_epsilon, mu_delta, std_delta = demand
    lb_L, ub_L, det = lead_times


    A = bill_of_materials(inst, comp, prod, min_use, max_use, seed, other)
    if inst is not None:
        comp = len(A); prod = len(A[0])
    else:
        pass
    
    C, H, pi = parametros(inst, comp, A, min_cost, max_cost, inv_factor, scenarios, seed)
    price, a, b, I0 = price_set(inst, a, b, lb_price, ub_price, step_price)
    mult = epsilon_ms(inst, stages, scenarios, branching, seed, lb_epsilon, ub_epsilon)
    add = delta_ms(inst, prod, stages, scenarios, branching, seed, mu_delta, std_delta)
    L = lead_times_ms(inst, comp, stages, scenarios, branching, seed, lb_L, ub_L, det)

    if I0 is None: 
        I0 = [0] * comp
    else: # input I0 is a scalar value
        if np.isscalar(a):
            I0 = [gp.quicksum((a - b * I0)*A[i][j] for j in range(prod)) for i in range(comp)]
        else:
            I0 = [gp.quicksum((a[0] - b[0][0] * I0)*A[i][j] for j in range(prod)) for i in range(comp)]
            # I0 = [gp.quicksum((a[j] - gp.quicksum(b[j][k]* I0[k] for k in range(prod)))*A[i][j] for j in range(prod)) for i in range(comp)]

    return C, H, pi, A, price, mult, add, L, a, b, I0


def uncertainty_analysis(vss_ms, evpi_ms, vss_ts, size, bom, costs, price, demand, lead_times, incumbent):
    
    inst, comp, prod, stages, scenarios, branching, seed, time_limit = size  
    C, H, pi, A, price, mult, add, L, a, b, I0 = extract_params(size, bom, costs, price, demand, lead_times)
    _,_, det = lead_times

    if inst is not None:
        comp = len(A); prod = len(A[0])

    if vss_ms:
        vss = VSS(seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, incumbent, I0)
        print(f"Value of Stochastic Solution (VSS): {vss:.4f}%")
    else:
        vss = -1
        pass

    if vss_ts:
        vss_2s = VSS_2S(seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, incumbent, I0)
        print(f"Value of Stochastic Solution - Two Stage (VSS_TS): {vss_2s:.4f}%")
        # seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, incumbent, I0
    else:
        vss_2s = -1
        pass

    if evpi_ms:
        evpi = EVPI(seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, incumbent, I0)
        print(f"Expected Value of Perfect Information (EVPI): {evpi:.4f}%")
    else:
        evpi = -1
        pass
    return vss, evpi, vss_2s