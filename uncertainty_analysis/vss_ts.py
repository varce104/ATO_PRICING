from models.twostage_dlt_slt import Recourse_problem
from models.multistage import Multistage_problem
import gurobipy as gp
from gurobipy import GRB
import numpy as np



def EEV_2S(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0):
    comp, prod = len(A), len(A[0])

    x_2s, w_2s = Recourse_problem(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, I0)
    if x_2s is None or w_2s is None:
        return -1
    
    m, x, w, y, I, A_out, D_term = Multistage_problem(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0)
    
    for i in range(comp):
        for t in range(time):
            for s in range(scenarios):
                x[i, t, s].LB = x_2s[i, t]
                x[i, t, s].UB = x_2s[i, t]

    for j in range(prod):
        for t in range(time):
            for s in range(scenarios):
                for p in range(len(price)):
                    val = 1.0 if w_2s[j, t, p] > 0.5 else 0.0
                    w[j, t, p, s].LB = val
                    w[j, t, p, s].UB = val

    m.setParam('OutputFlag', 0)
    m.optimize()
    return m.objVal if m.status == GRB.OPTIMAL else -np.inf


def VSS_2S(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, incumbent, I0):
    rp_val = incumbent 

    eev_val = EEV_2S(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0)
    if eev_val == -1:
        return -1
    
    if eev_val == -np.inf or rp_val == 0: return -1
    
    vss = rp_val - eev_val
    vss_pct = (vss / abs(rp_val)) * 100
    return vss_pct