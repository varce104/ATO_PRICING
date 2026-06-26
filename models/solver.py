from data.generator import epsilon_ms, delta_ms, lead_times_ms
from data.params import parametros, bill_of_materials, price_set

from models.multistage import Multistage_problem
from models.linealization_prop import MS_linear, TS_linear
from models.affine_funct_app import MS_linear_affine, MS_affine_cts
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

    # Manual increase of price set
    # price_factor = 10
    # price = np.round(np.linspace(lb_price*price_factor,ub_price*price_factor,15)).astype(int)
    # # Intercept of demand function. Must increase in same factor, otherwise demand allways <0 -> model unfeasible
    # a = a*price_factor 

    if I0 is None: 
        I0 = [0] * comp
    else: # input I0 is a scalar value
        I0 = [gp.quicksum((a - b * I0)*A[i][j] for j in range(prod)) for i in range(comp)]

    return C, H, pi, A, price, mult, add, L, a, b, I0


def solve(size, bom, costs, price_param, demand, lead_times, show):

    inst, comp, prod, stages, scenarios, branching, seed, time_limit = size  
    C, H, pi, A, price, mult, add, L, a, b, I0 = extract_params(size, bom, costs, price_param, demand, lead_times)
    _,_, det = lead_times
    show_var, lambda_app, lambda_benders, show_heatmap, show_boxplot, show_candlestick, vss_calc, evpi_calc, vss_ts_calc, Model, W_cts = show
    
    
    if inst is not None:
        comp = len(A); prod = len(A[0])
        print(f"\nInstance: {inst} | Model: {Model} | Components: {comp} | Products: {prod} | Stages: {stages} | Scenarios: {scenarios} | Iteration: {seed-4}")

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
        print(f"\n--- Construyendo modelo linealizado ---    w relajado: {W_cts}")
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
            m, x_vars, w_vars, y_vars, I_vars, A, D_term, rho, gamma, K_features = MS_linear_affine(
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
        m, obj, ex_time, iteration = Iter_policy(seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
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

    vss, evpi, vss_ts = uncertainty_analysis(vss_calc, evpi_calc, vss_ts_calc, size, bom, costs, price_param, demand, lead_times, incumbent)
    solutions = [seed, x_vars, w_vars, I_vars, y_vars, D_term, price, stages, scenarios, A, det, lambda_app, Model, rho, gamma, K_features]
    export(show_var, show_heatmap, show_boxplot, show_candlestick, solutions)


    return {"incumbent": incumbent, "bestbd": bestbd, "gap": gap, "time": opt_time, "vss": vss, "evpi": evpi, "vss_ts": vss_ts}