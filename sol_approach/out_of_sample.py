import numpy as np
from itertools import product
from models.linealization_prop import MS_linear
from data.phi_features import build_phi


def Affine_OOS_eval(oos_seed, stages, oos_scenarios, A, price, L_oos, L_det,
                    ypsilon_oos, delta_oos, a, b, C, H, pi_oos, branching_oos, I0, 
                    rho_opt, Gamma_opt, phi_mode="eps_delta"):
    """
    Somete la política afín in-sample a un nuevo árbol estocástico out-of-sample.
    """
    comp, prod, pr = len(A), len(A[0]), len(price)

    # 1. Construir la matriz de características para el nuevo árbol OOS
    phi_oos, K_feat = build_phi(phi_mode, stages, oos_scenarios, prod, comp, 
                                ypsilon_oos, delta_oos, L_oos, L_det)

    # 2. Calcular los pesos continuos OOS usando la regla de decisión paramétrica
    lam_oos = np.zeros((prod, stages, pr, oos_scenarios))
    w_bin_oos = np.zeros((prod, stages, pr, oos_scenarios))

    for j, t, s in product(range(prod), range(stages), range(oos_scenarios)):
        for p in range(pr):
            # Proyección del intercepto y las pendientes sobre la nueva historia observada
            lam_oos[j, t, p, s] = rho_opt[j, t, p] + np.dot(Gamma_opt[j, t, p, :], phi_oos[s][t][:])
            
        # 3. Binarización determinista (argmax)
        best_p = int(np.argmax(lam_oos[j, t, :, s]))
        w_bin_oos[j, t, best_p, s] = 1.0

    # 4. Construir el modelo multietapa extensivo para el árbol OOS
    m_oos, x_vars, w_vars, y_vars, I_vars, A_out, D_term = MS_linear(
        oos_seed, stages, oos_scenarios, A, price, L_oos, L_det,
        ypsilon_oos, delta_oos, a, b, C, H, pi_oos, branching_oos, I0)

    # 5. Fijar las decisiones de pricing a la política evaluada
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(oos_scenarios)):
        val = float(w_bin_oos[j, t, p, s])
        w_vars[j, t, p, s].LB = val
        w_vars[j, t, p, s].UB = val

    return m_oos, x_vars, w_vars, y_vars, I_vars, A_out, D_term