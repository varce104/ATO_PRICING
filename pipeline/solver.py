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
    show_heatmap = cfg.run.show_heatmap
    show_boxplot = cfg.run.show_boxplot
    show_candlestick = cfg.run.show_candlestick
    show_kpis = cfg.run.show_kpis
    
    if inst is not None:
        comp = len(A); prod = len(A[0])
        print(f"\nInstance: {seed-4} | Model: {Model} | seed: {seed} | Components: {comp} | Products: {prod} | Stages: {stages} | Scenarios: {scenarios} ")

    solve_time = 0
    rho=0; gamma=0; K_features=0
    iterations=None

    if Model == "MS": #Non-linear model
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = Multistage_problem(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        if lambda_app:
            w_sim = pd.read_excel(f"var_results/MS_lambda_app_inst{seed}.xlsx", sheet_name='W_sol', index_col=[0, 1, 2])
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


    elif Model == "MS_linear_affine":
        if W_cts:
            m, x_vars, w_vars, y_vars, I_vars, A, D_term, rho, gamma, K_features = MS_affine_cts(
                seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
                phi_mode=cfg.run.phi_mode)
        else:
            m, x_vars, w_vars, y_vars, y_bar, I_vars, A, D_term, rho, gamma, K_features = MS_linear_affine(
                seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
                phi_mode=cfg.run.phi_mode)

    elif Model == "TS_linear_affine":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, _, _ = TS_linear_affine(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, I0,
            phi_mode=cfg.run.phi_mode)


    elif Model == "Affine_eval":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Affine_eval(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
            phi_mode=cfg.run.phi_mode)

    elif Model == "Relaxed_eval":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Relaxed_eval(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        # Relaxed_eval no usa phi (resuelve MS_linear relajado en w, no la política afín) — sin cambios


    elif Model == "Iterative_heuristic":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = Iter_policy(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
            max_iter=cfg.iter, phi_mode=cfg.run.phi_mode)

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

    solutions = [seed, x_vars, w_vars, I_vars, y_vars, D_term, price, stages, scenarios, A, det,
                 lambda_app, Model, rho, gamma, K_features, mult, add, a, b, pi, I0]

    results = {"incumbent": incumbent, "bestbd": bestbd, "gap": gap, "time": opt_time, "iterations (IH)": iterations, "vss": vss, "evpi": evpi, "vss_ts": vss_ts}

    if show_kpis:
        results["Fill Rate"], results["Utilization Rate"], results["Inventory Service Rate"], results["Weighted Avg Price"] = export(
            show_var, show_heatmap, show_boxplot, show_candlestick, show_kpis, solutions)
        return results
    else:
        return results
