import numpy as np
import random



# Oh_et_al_1
# 3x2
# BoM = [[1,0], [1,1], [0,1]]
# Price set: [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# Price function: D(p) = 100 - 1.6*p
# Costs: C = [5, 5, 45]
# Inventory Costs: I = [4, 4, 36]

# Oh_et_al_2
# 5x4
# BoM: [[1,1,0,0], [2,1,1,0], [1,1,1,0], [0,0,1,0], [0,0,0,1]]
# Price set: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Price function: D(p) = 50 - 0.8*p
# Costs: C = [15,15,10,20,15]
# Inventory Costs: I = C * 0.15

# Oh_et_al_3
# 11x11
# BoM: Every product uses random 5 components from the 11 total
# Price set: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Price function: D(p) = 50 - 0.8*p
# Costs: C = 
# Inventory Costs: I = C * 0.15

def bill_of_materials(inst, comp, prod, min_use, max_use, seed, other): 
    seed = 6
    random.seed(seed)
    np.random.seed(seed)
    A = np.zeros((comp, prod), dtype=int)

    if inst == "Oh_et_al_1": # 3x2
        A = [[1,0],
             [1,1],
             [0,1]]

    elif inst == "Oh_et_al_2": # 5x4
        A = [[1,1,0,0],
             [2,1,1,0],
             [1,1,1,0],
             [0,0,1,0],
             [0,0,0,1]]

    elif inst == "Oh_et_al_3": # 11x11
        comp = 11; prod = 11; min_use = 3; max_use = 5
        A = np.zeros((comp, prod), dtype=int)       
        for j in range(prod):
            k_j = np.random.randint(min_use, max_use + 1)
            chosen_components = np.random.choice(
                comp,
                size=k_j,
                replace=False)
            A[chosen_components, j] = 1
    else:
        for j in range(prod):
            k_j = np.random.randint(min_use, max_use + 1)
            chosen_components = np.random.choice(
                comp,
                size=k_j,
                replace=False) # False Only for 5x4, True otherwise
            A[chosen_components, j] = 1

        if other:
            A = [[1,0],
                 [1,1],
                 [0,1]]
            #A = [[1,1],[0,1]]        

    return A

def price_set(inst, a, b, inf, sup, step):
    if inst == "Oh_et_al_1":
        inf = 15; sup = 60; a = 100; b = 1.6
    
    elif inst == "Oh_et_al_2":
        inf = 5; sup = 50; a = 50; b = 0.8

    elif inst == "Oh_et_al_3":
        inf = 5; sup = 50; a = 50; b = 0.8
        # inf = 30; sup = 80; a = 200; b = 2.0

    else:
        pass
    I0 = np.max([inf, sup])
    return list(range(inf, sup + 1, step)), a, b, I0

def parametros(inst, comp, A, min_cost, max_cost, inv_factor, scenarios, seed, prob=True):
    seed = 6
    random.seed(seed)
    np.random.seed(seed)
    
    if inst == "Oh_et_al_1":
        C = [5, 5, 45]; I = [4, 4, 36]

    elif inst == "Oh_et_al_2":
        C = [15,15,10,20,15]
        I = [x * 0.15 for x in C]

    elif inst == "Oh_et_al_3":
        total_cost = 55; inv_factor = 0.15
        k = int(np.sum(np.array(A)[:, 0]))  # k es igual para todas las columnas
        c = round(total_cost / k)
        C = [c] * comp
        I = [x * inv_factor for x in C]

    else:
        C = [random.randrange(min_cost, max_cost+1, 5) for i in range(comp)]
        I = [x * inv_factor for x in C]

    if prob:
        prob_scenarios = [1/scenarios for s in range(scenarios)] # probabilidad uniforme
        return C, I, prob_scenarios
    else:
        return C, I
    
# A = bill_of_materials( "None", 5, 4, 3, 4, 6, False)
# C, I,_ = parametros( "None", 5, A, 5, 25, 0.2, 128, 6)
# print(A)
# print(C)
# print(I)