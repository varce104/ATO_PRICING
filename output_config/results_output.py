from output_config.var_export import export_solution_to_excel, extract_solution_arrays
from output_config.graphs.heatmap import plot_instance_decisions
from output_config.graphs.boxplot import plot_boxplot
from output_config.graphs.candlestick import plot_candlestick
from output_config.lambda_export import export_solution_to_excel_affine, export_affine_params_to_excel
import numpy as np


def export(show_sol, show_heatmap, show_boxplot, show_candlestick, vals):
    seed, x_vars, w_vars, I_vars, y_vars, D_term, price, time, scenarios, A, det, lambda_app, Model, rho, gamma, K_features = vals

    if lambda_app and Model in ("MS_linear_affine", "TS_linear_affine"):
        export_solution_to_excel_affine(f"var_results/MS_lambda_app_inst{seed}.xlsx", w_vars, time, scenarios, len(price), A, Model)
        export_affine_params_to_excel(f"var_results/MS_affine_params_inst{seed}.xlsx",rho, gamma, len(A[0]), time, len(price), K_features)
        return None
    else:
        pass

    
    if show_sol:
        x_val, price_eff, I_val, y_val, d_val = extract_solution_arrays(x_vars, w_vars, I_vars, y_vars, D_term, price, len(A), len(A[0]), time, scenarios, pr=len(price))
        if det:
            export_solution_to_excel(f"var_results/MS_DL_inst{seed}.xlsx", x_val, price_eff, y_val, I_val, d_val, time, scenarios, A)
        else:
            export_solution_to_excel(f"var_results/MS_SL_inst{seed}.xlsx", x_val, price_eff, y_val, I_val, d_val, time, scenarios, A)
    else:
        pass


    if show_heatmap:
        x_avg = np.mean(x_val, axis=2)
        p_avg = np.mean(price_eff, axis=2)
        I_avg = np.mean(I_val, axis=2)
        y_avg = np.mean(y_val, axis=2)
        d_avg = np.mean(d_val, axis=2)
        plot_instance_decisions(x_avg, p_avg, I_avg, y_avg, d_avg)
    else:
        pass

    if show_boxplot:
        plot_boxplot(x_val, price_eff, I_val, y_val, d_val)
    else:
        pass

    if show_candlestick:
        x_val, price_eff, I_val, y_val, d_val = extract_solution_arrays(x_vars, w_vars, I_vars, y_vars, D_term, price, len(A), len(A[0]), time, scenarios, pr=len(price))
        plot_candlestick(x_val, price_eff, I_val, y_val, d_val)
    else:
        pass
