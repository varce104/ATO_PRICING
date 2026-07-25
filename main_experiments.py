import copy
import math
from data.config import (ProblemSize, BomConfig, CostConfig, PriceConfig,
                          DemandConfig, LeadTimeConfig, RunConfig, ExperimentConfig)
from pipeline.experiments import run_parameter, apply_comp, apply_prod, apply_branching, apply_phi_mode

# --- Instancia base (idéntica a la de main.py) ---
comp, prod, stages = 5, 4, 8
branching = [2]*7
scenarios = math.prod(branching)

size = ProblemSize("None", comp, prod, stages, scenarios, branching, seed=5, time_limit=900)
bom = BomConfig(min_use=3, max_use=4, other=False)
costs = CostConfig(min_cost=5, max_cost=25, inv_factor=0.2, I0=60)
price = PriceConfig(lb_price=15, ub_price=60, step_price=5)
demand = DemandConfig(a=100, b=1.6, lb_epsilon=0.7, ub_epsilon=1.3, mu_delta=0, std_delta=2)
lead_times = LeadTimeConfig(lb_L=1, ub_L=2, det=False)

run = RunConfig(Model="MS_linear", show_fulfillment=False)

base_cfg = ExperimentConfig(size, bom, costs, price, demand, lead_times, run, iter=3)

# --- Eje 1: número de componentes ---
run_parameter(base_cfg, "comp", [4, 5, 6, 7, 8, 9, 10], apply_comp)

# --- Eje 2: número de productos ---
run_parameter(base_cfg, "prod", [3, 4, 5, 6, 7, 8, 9], apply_prod)

# --- Eje 3: branching (definir tus propias formas de árbol aquí) ---
branching_options = [
    [2,2,2,2,2,2,2],    
    [5,5,5,2,1,1,1],
    [10,5,2,2,1,1,1],
    [20,5,2,1,1,1,1],
    [50,2,2,1,1,1,1],
    [125,2,1,1,1,1,1],
]
run_parameter(base_cfg, "branching", branching_options, apply_branching)

# --- Eje 4: modo de phi (para modelos afines) ---
affine_cfg = copy.deepcopy(base_cfg)
affine_cfg.run = RunConfig(Model="Affine_eval",)
run_parameter(affine_cfg, "phi_mode", ["eps", "eps_delta", "eps_lt", "eps_delta_lt"], apply_phi_mode)


import pandas as pd
import itertools
import time

# 1. Definir los modelos a evaluar
# Aquí listarías las referencias a tus modelos (ej. multistage, twostage, etc.)
modelos_a_ejecutar = ['multistage', 'twostage_dlt_slt', 'affine_funct_app']

# 2. Definir los parámetros generales y las instancias a variar
# Los valores en listas representan las variaciones que quieres probar
parametros_instancias = {
    'capacidad_inventario': [100, 200, 500],
    'tasa_demanda': [10, 15],
    'costo_penalizacion': [0.5, 1.0]
}

# 3. Generar todas las combinaciones posibles de parámetros
nombres_params, valores_params = zip(*parametros_instancias.items())
combinaciones = [dict(zip(nombres_params, v)) for v in itertools.product(*valores_params)]

def ejecutar_experimentos(modelos, instancias):
    resultados_totales = []
    
    # 4. Bucle anidado: Iterar sobre cada modelo y cada combinación de parámetros
    for nombre_modelo in modelos:
        print(f"--- Iniciando corridas para el modelo: {nombre_modelo} ---")
        
        for i, instancia in enumerate(instancias):
            print(f"  Ejecutando instancia {i+1}/{len(instancias)}: {instancia}")
            
            inicio = time.time()
            
            # ------------------------------------------------------------------
            # AQUÍ LLAMAS A TU PIPELINE/SOLVER
            # Ejemplo conceptual:
            # modelo_obj = instanciar_modelo(nombre_modelo, parametros_generales)
            # output = modelo_obj.resolver(instancia)
            # ------------------------------------------------------------------
            
            # Simulamos el diccionario de salida que te daría tu solver
            output_simulado = {
                'funcion_objetivo': 15420.5, 
                'gap_optimizacion': 0.01,
                'estado_solver': 'optimal'
            }
            
            fin = time.time()
            
            # 5. Consolidar la información de la corrida
            registro = {
                'Modelo': nombre_modelo,
                **instancia,                  # Desempaqueta los parámetros usados
                **output_simulado,            # Desempaqueta los resultados del modelo
                'Tiempo_Ejecucion_Seg': round(fin - inicio, 2)
            }
            
            resultados_totales.append(registro)
            
    return resultados_totales

# 6. Ejecutar y exportar
resultados = ejecutar_experimentos(modelos_a_ejecutar, combinaciones)

# Convertir la lista de diccionarios a un DataFrame y exportar a un único Excel
df_resultados = pd.DataFrame(resultados)

# Puedes guardar todo en una sola hoja, o usar pd.ExcelWriter para separar por modelo
nombre_archivo = 'resultados_experimentos.xlsx'
df_resultados.to_excel(nombre_archivo, index=False)
print(f"\n¡Todos los experimentos finalizados! Resultados exportados en: {nombre_archivo}")