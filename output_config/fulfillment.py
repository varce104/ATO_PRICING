import numpy as np


def demand_fulfillment_old(y_val, price_eff, ypsilon, delta, a, b, pi):
    """
    Calcula el nivel de demanda satisfecha (fill rate) a partir de arrays ya extraídos.

    y_val, price_eff : arrays (prod, time, scenarios), salida de extract_solution_arrays
    ypsilon           : mult[s][t] (componente multiplicativo, shape scenarios x time)
    delta             : add[s][j][t] (componente aditivo, shape scenarios x prod x time)
    a, b              : parámetros de la función de demanda lineal
    pi                : probabilidades por escenario

    La demanda efectiva NO se recalcula desde D_term: su indexación cambia entre
    modelos (D_term[j,t,s] en Multistage_problem vs D_term[j,t,p,s] en el resto),
    lo que hace que extract_solution_arrays falle fuera del modelo "MS". En su
    lugar se reconstruye D_eff desde el precio efectivamente asignado (price_eff),
    válido para cualquier modelo cuyo price_eff se haya podido extraer.

    Retorna:
        fill_jts    (prod, time, scenarios) en %  -> detalle por producto
        fill_ts     (time, scenarios) en %         -> agregado por producto, listo para candlestick
        fill_t      (time,) en %                   -> esperado por período (ponderado por pi)
        fill_global float en %                     -> esperado total
    """

    prod, time, scenarios = y_val.shape
    ypsilon = np.array(ypsilon)          # (scenarios, time)
    pi = np.array(pi)

    D_eff = np.zeros((prod, time, scenarios))
    for j in range(prod):
        delta_j = np.array([[delta[s][j][t] for t in range(time)] for s in range(scenarios)])  # (scenarios, time)
        D_eff[j] = (ypsilon * (a - b * price_eff[j].T) + delta_j).T  # -> (time, scenarios)

    D_eff = np.clip(D_eff, 0.0, None)  # demanda "negativa" (precio muy alto) no tiene sentido como denominador

    with np.errstate(divide='ignore', invalid='ignore'):
        fill_jts = np.where(D_eff > 1e-9, y_val / D_eff, 1.0)  # sin demanda -> 100% satisfecho por convención
    fill_jts = np.clip(fill_jts, 0.0, 1.0)

    y_sum_ts = y_val.sum(axis=0)
    d_sum_ts = D_eff.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fill_ts = np.where(d_sum_ts > 1e-9, y_sum_ts / d_sum_ts, 1.0)

    y_sum_t = (y_sum_ts * pi).sum(axis=1)
    d_sum_t = (d_sum_ts * pi).sum(axis=1)
    fill_t = np.where(d_sum_t > 1e-9, y_sum_t / d_sum_t, 1.0)

    fill_global = float(y_sum_t.sum() / d_sum_t.sum()) if d_sum_t.sum() > 1e-9 else 1.0

    return fill_jts, fill_ts, fill_t, fill_global

def reconstruct_demand(price_eff, ypsilon, delta, a, b):
    """
    Reconstruye D_eff (prod, time, scenarios) desde el precio efectivamente
    asignado, evitando depender de D_term (indexación inconsistente entre modelos).
    """
    prod, time, scenarios = price_eff.shape
    ypsilon = np.array(ypsilon)
    D_eff = np.zeros((prod, time, scenarios))
    for j in range(prod):
        delta_j = np.array([[delta[s][j][t] for t in range(time)] for s in range(scenarios)])
        D_eff[j] = (ypsilon * (a - b * price_eff[j].T) + delta_j).T
    return np.clip(D_eff, 0.0, None)


def demand_fulfillment(y_val, price_eff, ypsilon, delta, a, b, pi):
    prod, time, scenarios = y_val.shape
    pi = np.array(pi)

    D_eff = reconstruct_demand(price_eff, ypsilon, delta, a, b)

    with np.errstate(divide='ignore', invalid='ignore'):
        fill_jts = np.where(D_eff > 1e-9, y_val / D_eff, 1.0)
    fill_jts = np.clip(fill_jts, 0.0, 1.0)

    y_sum_ts = y_val.sum(axis=0)
    d_sum_ts = D_eff.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fill_ts = np.where(d_sum_ts > 1e-9, y_sum_ts / d_sum_ts, 1.0)

    y_sum_t = (y_sum_ts * pi).sum(axis=1)
    d_sum_t = (d_sum_ts * pi).sum(axis=1)
    fill_t = np.where(d_sum_t > 1e-9, y_sum_t / d_sum_t, 1.0)

    fill_global = float(y_sum_t.sum() / d_sum_t.sum()) if d_sum_t.sum() > 1e-9 else 1.0

    return fill_jts, fill_ts, fill_t, fill_global