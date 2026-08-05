from output_config.var_export import export_solution_to_excel, extract_solution_arrays, export_long_format_csv

from output_config.graphs.heatmap import plot_instance_decisions
from output_config.graphs.boxplot import plot_boxplot
from output_config.graphs.candlestick import plot_candlestick_decisions

from output_config.lambda_export import export_solution_to_excel_affine, export_affine_params_to_excel
from output_config.fulfillment import demand_fulfillment, reconstruct_demand
from output_config.kpi_performance import calculate_global_kpis
import numpy as np


def export(show_sol, show_heatmap, show_boxplot, show_candlestick, show_kpis, solutions):
    (seed, x_vars, w_vars, I_vars, y_vars, D_term, price, time, scenarios, A, det,
     lambda_app, Model, rho, gamma, K_features, mult, add, a, b, pi, I0) = solutions

    if lambda_app and Model in ("MS_linear_affine", "TS_linear_affine"):
        export_solution_to_excel_affine(f"var_results/MS_lambda_app_inst{seed}.xlsx", w_vars, time, scenarios, len(price), A, Model)
        export_affine_params_to_excel(f"var_results/MS_affine_params_inst{seed}.xlsx", rho, gamma, len(A[0]), time, len(price), K_features)
        return None

    need_extraction = show_sol or show_heatmap or show_boxplot or show_candlestick or show_kpis

    if need_extraction:
        x_val, price_eff, I_val, y_val = extract_solution_arrays(
            x_vars, w_vars, I_vars, y_vars, D_term, price, len(A), len(A[0]), time, scenarios, pr=len(price))
        D_val = reconstruct_demand(price_eff, mult, add, a, b)   # <-- reemplaza a D_term en gráficos

    if show_sol:
        # Mantiene tu exportación original (para lectura manual)
        # if det:
        #     export_solution_to_excel(f"var_results/MS_DL_inst{seed}.xlsx", x_val, price_eff, y_val, I_val, D_val, time, scenarios, A)
        # else:
        #     export_solution_to_excel(f"var_results/MS_SL_inst{seed}.xlsx", x_val, price_eff, y_val, I_val, D_val, time, scenarios, A)
        # NUEVO: Genera el dataset consolidado para comparar modelos
        export_long_format_csv(filename="var_results/concat_sol.csv",Model=Model,seed=seed,x_val=x_val,p_eff=price_eff,y_val=y_val,
            I_val=I_val,d_val=D_val,time=time,scenarios=scenarios,A=A)

    if show_heatmap:
        x_avg = np.mean(x_val, axis=2)
        p_avg = np.mean(price_eff, axis=2)
        I_avg = np.mean(I_val, axis=2)
        y_avg = np.mean(y_val, axis=2)
        d_avg = np.mean(D_val, axis=2)
        plot_instance_decisions(x_avg, p_avg, I_avg, y_avg, d_avg, Model, seed)

    if show_boxplot:
        plot_boxplot(x_val, price_eff, I_val, y_val, D_val)

    if show_candlestick:
        plot_candlestick_decisions(x_val, price_eff, I_val, y_val, D_val, Model, seed)

    if show_kpis:
        # Llamada directa pasando los arrays Numpy y parámetros desempaquetados de 'solutions'
        FR, UR, ISR, VWAP = calculate_global_kpis(x_val, price_eff, y_val, I_val, A, mult, add, a, b, pi, I0)

    return FR, UR, ISR, VWAP
