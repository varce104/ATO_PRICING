from models.multistage import Multistage_problem

import numpy as np
import gurobipy as gp
from gurobipy import GRB



def Wait_and_see(seed, time, s, A, price, L, L_det, ypsilon, delta, a, b, C, H, I0):
    comp, prod = len(A), len(A[0])
    
    y_s = [ypsilon[s]] 
    d_s = [delta[s]]
    pi_s = [1.0]
    branch_ws = [1] * time 
    
    if L_det:
        L_s = L
        m, _, _, _, _, _, _ = Multistage_problem(
            seed, time, 1, A, price, L_s, L_det, y_s, d_s, a, b, C, H, pi_s, branch_ws, I0)
    else:
        L_s = [[[L[i][t][s]] for t in range(time)] for i in range(comp)]
        m, _, _, _, _, _, _ = Multistage_problem(
            seed, time, 1, A, price, L_s, L_det, y_s, d_s, a, b, C, H, pi_s, branch_ws, I0)

    m.setParam('OutputFlag', 0)
    m.optimize()
    
    return m.objVal if m.status == GRB.OPTIMAL else 0



def EVPI(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, incumbent, I0):
    
    rp_val = incumbent 
    
    ws_weighted_sum = 0
    for s in range(scenarios):
        val_s = Wait_and_see(seed, time, s, A, price, L, L_det, ypsilon, delta, a, b, C, H, I0)
        ws_weighted_sum += pi[s] * val_s

    if rp_val == 0: return -1
    
    evpi = ws_weighted_sum - rp_val
    evpi_pct = (evpi / abs(rp_val)) * 100
    
    return evpi_pct