from data.generator import epsilon_ms, delta_ms, lead_times_ms
from data.params import parametros, bill_of_materials, price_set
from data.extract import extract_params

from models.multistage import Multistage_problem
from models.linealization_prop import MS_linear, TS_linear
from sol_approach.affine_funct_app import MS_linear_affine, MS_affine_cts
from sol_approach.price_app_eval import Affine_eval, Relaxed_eval
from sol_approach.twostage_affine import TS_linear_affine
from sol_approach.iterative_heuristic import Iter_policy
from uncertainty_analysis.sto_computation import uncertainty_analysis

from output_config.results_output import export
from output_config.lambda_export import fix_w_from_lambda

import pandas as pd
import numpy as np
from itertools import product
import gurobipy as gp
from gurobipy import GRB



def solve(cfg):
    # Único punto donde se toca cfg — el resto de la función queda igual que antes
    C, H, pi, A, price, mult, add, L, a, b, I0 = extract_params(cfg)

    inst, comp, prod = cfg.size.inst, cfg.size.comp, cfg.size.prod
    stages, scenarios, branching = cfg.size.stages, cfg.size.scenarios, cfg.size.branching
    seed, time_limit = cfg.size.seed, cfg.size.time_limit
    det = cfg.lead_times.det

    Model = cfg.run.Model
    W_cts = cfg.run.W_cts
    show_var = cfg.run.show_var
    lambda_app = cfg.run.lambda_app
    lambda_benders = cfg.run.lambda_benders
    show_heatmap = cfg.run.show_heatmap
    show_boxplot = cfg.run.show_boxplot
    show_candlestick = cfg.run.show_candlestick
    vss_calc = cfg.run.vss_calc
    evpi_calc = cfg.run.evpi_calc
    vss_ts_calc = cfg.run.vss_ts_calc
    
    
    if inst is not None:
        comp = len(A); prod = len(A[0])
        print(f"\nInstance: {seed-4} | Model: {Model} | seed: {seed} | Components: {comp} | Products: {prod} | Stages: {stages} | Scenarios: {scenarios} ")

    solve_time = 0
    rho=0; gamma=0; K_features=0

    if Model == "MS": #Non-linear model
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = Multistage_problem(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        if lambda_app:
            w_sim = pd.read_excel(f"var_results/MS_lambda_app_inst{seed}.xlsx", sheet_name='W_sol', index_col=[0, 1, 2])
            fix_w_from_lambda(m, w_vars, w_sim, prod, stages, len(price), scenarios)
        if lambda_benders:
            w_sim = pd.read_excel(f"var_results/MS_benders_inst{seed}.xlsx", sheet_name='W_sol', index_col=[0, 1, 2])
            fix_w_from_lambda(m, w_vars, w_sim, prod, stages, len(price), scenarios)
            

    elif Model == "MS_linear": # Linear model (base model)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = MS_linear(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, W_cts)
        if lambda_app:
            w_sim = pd.read_excel(f"var_results/MS_lambda_app_inst{seed}.xlsx", sheet_name='W_sol', index_col=[0, 1, 2])
            fix_w_from_lambda(m, w_vars, w_sim, prod, stages, len(price), scenarios)

    elif Model == "TS_linear": # Two stage version of MS_linear
        print(f"\n--- Construyendo modelo linealizado ---    w relajado: {W_cts}")
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = MS_linear(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, W_cts)
        if lambda_app:
            w_sim = pd.read_excel(f"var_results/MS_lambda_app_inst{seed}.xlsx", sheet_name='W_sol', index_col=[0, 1, 2])
            fix_w_from_lambda(m, w_vars, w_sim, prod, stages, len(price), scenarios)


    elif Model == "MS_linear_affine": # Pricing approximation via affine functions (LP model)
        if W_cts:
            m, x_vars, w_vars, y_vars, I_vars, A, D_term, rho, gamma, K_features = MS_affine_cts(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        else:    
            m, x_vars, w_vars, y_vars, y_bar, I_vars, A, D_term, rho, gamma, K_features = MS_linear_affine(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        
    elif Model == "TS_linear_affine": # Two stage version of MS_linear_affine
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, _, _ = TS_linear_affine(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, I0)     


    elif Model == "Affine_eval": # Approximation and evaluation of MS_linear_affine pricing policy into MS_linear (accounts for solving these two)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Affine_eval(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        if m == None:
            return {"incumbent": None, "bestbd": None, "gap": None, "time": None, "vss": None, "evpi": None, "vss_ts": None}

    elif Model == "Relaxed_eval": # Approximation and evaluation of relaxed MS_linear pricing policy into MS_linear (accounts for solving these two)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Relaxed_eval(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
    

    elif Model == "Iterative_heuristic": # Iteration between Pricing only model and ATO model.
        m, obj, ex_time, iteration = Iter_policy(seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, max_iter=cfg.iter)
        if m == None:
            return {"incumbent": None, "bestbd": None, "gap": None, "time": None, "vss": None, "evpi": None, "vss_ts": None}
        else:
            return {"incumbent": obj, "bestbd": obj, "gap": ((obj-obj)/obj*100), "time": ex_time, "iteration": iteration}
        
#=====================================================================================================================================
    else:
        print("\nWrong input, try again...")
        return None
#=====================================================================================================================================

#=====================================================================================================================================
#   Gurobi Parameters:
    m.setParam('TimeLimit', time_limit)
    m.setParam('OutputFlag', 1)
    # m.setParam('Method', 1)
    m.setParam('BarHomogeneous', 1)
    # m.setParam("MIPGap", 5e-4) # Gap tol: 0.05% // Gurobi base tol: 0.01%/1e-4
    # m.Params.NonConvex = 2
#=====================================================================================================================================
    m.optimize()

    if m.status == GRB.OPTIMAL:
        incumbent = m.objVal
        bestbd = m.objBound
        gap = 0.0

    elif m.status == GRB.TIME_LIMIT:
        if m.SolCount > 0:
            incumbent = m.objVal
            bestbd = m.objBound
            gap = m.MIPGap
        else:
            print("====== No solution available ======")
            incumbent = None
            bestbd = m.objBound
            gap = None

    else:
        incumbent = None
        bestbd = None
        gap = None

    opt_time = m.Runtime + solve_time
    vss, evpi, vss_ts = uncertainty_analysis(cfg, incumbent)

    solutions = [seed, x_vars, w_vars, I_vars, y_vars, D_term, price, stages, scenarios, A, det, lambda_app, Model, rho, gamma, K_features]
    export(show_var, show_heatmap, show_boxplot, show_candlestick, solutions)

    return {"incumbent": incumbent, "bestbd": bestbd, "gap": gap, "time": opt_time, "vss": vss, "evpi": evpi, "vss_ts": vss_ts}