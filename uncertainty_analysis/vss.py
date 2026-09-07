import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random

def build_model_for_metric(cfg, seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0):
    if cfg.run.Model in ["MS_linear", "MS_linear_affine", "AF_EVAL", "REL_EVAL", "IH"]:
        from models.linealization_prop import MS_linear
        return MS_linear(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0, cfg.run.W_cts)
    else:
        from models.multistage import Multistage_problem
        return Multistage_problem(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0)

def Expected_value(cfg, seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, I0):
    random.seed(seed)
    comp, prod = len(A), len(A[0])
    
    y_mean = [[np.mean([ypsilon[s][t] for s in range(scenarios)]) for t in range(time)]]
    d_mean = [[[np.mean([delta[s][j][t] for s in range(scenarios)]) for t in range(time)] for j in range(prod)]]
    
    pi_ev = [1.0]
    branch_ev = [1] * time 
    
    if L_det:
        L_mean = L
    else:
        L_mean = [[[int(np.mean(L[i][t]))] * scenarios  for t in range(time)] for i in range(comp)]
        
    m, x, w, _, _, _, _ = build_model_for_metric(
        cfg, seed, time, 1, A, price, L_mean, L_det, y_mean, d_mean, a, b, C, H, pi_ev, branch_ev, I0)

    m.setParam('OutputFlag', 0)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        x_val = {(i, t): x[i, t, 0].X for i in range(comp) for t in range(time)}
        w_val = {(j, t, p): w[j, t, p, 0].X for j in range(prod) for t in range(time) for p in range(len(price))}
        return x_val, w_val
    else:
        return None, None

def EEV(cfg, seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0):
    x_ev, w_ev = Expected_value(cfg, seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, I0)
    if x_ev is None: return -np.inf

    m, x, w, _, _, _, _ = build_model_for_metric(
        cfg, seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0)

    comp, prod = len(A), len(A[0])
    
    # Fijar variables de aquí-y-ahora (dependiendo si es bi-etapa o multietapa puro)
    for i in range(comp):
        for t in range(time):
            for s in range(scenarios):
                x[i, t, s].LB = x_ev[i, t]
                x[i, t, s].UB = x_ev[i, t]

    for j in range(prod):
        for t in range(time):
            for s in range(scenarios):
                for p in range(len(price)):
                    val = 1.0 if w_ev[j, t, p] > 0.5 else 0.0
                    w[j, t, p, s].LB = val
                    w[j, t, p, s].UB = val

    m.setParam('OutputFlag', 0)
    m.optimize()
    return m.objVal if m.status == GRB.OPTIMAL else -np.inf

def VSS(cfg, seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, incumbent, I0):
    rp_val = incumbent 
    eev_val = EEV(cfg, seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0)
    
    if eev_val == -np.inf or rp_val == 0 or rp_val is None: return -1
    return ((rp_val - eev_val) / abs(rp_val)) * 100