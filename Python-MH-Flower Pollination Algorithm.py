############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Flower Pollination Algorithm

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Flower_Pollination_Algorithm, File: Python-MH-Flower Pollination Algorithm.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Flower_Pollination_Algorithm>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import math
import random
import os

# Function: Initialize Variables
def initial_position(flowers = 3, min_values = [-5,-5], max_values = [5,5]):
    position = pd.DataFrame(np.zeros((flowers, len(min_values))))
    position['Fitness'] = 0.0
    for i in range(0, flowers):
        for j in range(0, len(min_values)):
             position.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        position.iloc[i,-1] = target_function(position.iloc[i,0:position.shape[1]-1])
    return position

# Function: Levy Distribution
def levy_flight(mean):
	x1 = math.sin((mean - 1.0)*(random.uniform(-0.5*math.pi, 0.5*math.pi)) )/(math.pow(math.cos((random.uniform(-0.5*math.pi, 0.5*math.pi))), (1.0/(mean - 1.0))))
	x2 = math.pow((math.cos((2.0 - mean)*(random.uniform(-0.5*math.pi, 0.5*math.pi)))/(-math.log(random.uniform(0.0, 1.0)))), ((2.0 - mean)/(mean - 1.0)))
	return x1*x2

# Function: Global Pollination
def pollination_global(position, best_global, flower = 0, gama = 0.5, lamb = 1.4, min_values = [-5,-5], max_values = [5,5]):
    x = best_global.copy(deep = True)
    for j in range(0, len(min_values)):
        x[j] = position.iloc[flower, j]  + gama*levy_flight(lamb)*(position.iloc[flower, j] - best_global[j])
        if (x[j] > max_values[j]):
            x[j] = max_values[j]
        elif(x[j] < min_values[j]):
            x[j] = min_values[j]
    x[-1]  = target_function(x[0:len(min_values)])
    return x

# Function: Local Pollination
def pollination_local(position, best_global, flower = 0, nb_flower_1 = 0, nb_flower_2 = 1, min_values = [-5,-5], max_values = [5,5]):
    x = best_global.copy(deep = True)
    for j in range(0, len(min_values)):
        r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        x[j] = position.iloc[flower, j]  + r*(position.iloc[nb_flower_1, j] - position.iloc[nb_flower_2, j])
        if (x[j] > max_values[j]):
            x[j] = max_values[j]
        elif(x[j] < min_values[j]):
            x[j] = min_values[j]
    x[-1]  = target_function(x[0:len(min_values)])
    return x

# FPA Function.
def flower_pollination_algorithm(flowers = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, gama = 0.5, lamb = 1.4, p = 0.8):    
    count = 0
    position = initial_position(flowers = flowers, min_values = min_values, max_values = max_values)
    best_global = position.iloc[position['Fitness'].idxmin(),:].copy(deep = True)
    x = best_global.copy(deep = True)
    
    while (count <= iterations):
        print("Iteration = ", count, " of ", iterations)
        for i in range(0, position.shape[0]):
            nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
            nb_flower_2 = int(np.random.randint(position.shape[0], size = 1))
            while nb_flower_1 == nb_flower_2:
                nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
            r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (r < p):
                x = pollination_global(position, best_global, flower = i, gama = gama, lamb = lamb, min_values = min_values, max_values = max_values)
            else:
                x = pollination_local(position, best_global, flower = i, nb_flower_1 = nb_flower_1, nb_flower_2 = nb_flower_2, min_values = min_values, max_values = max_values)
                    
            if (x[-1] <= position.iloc[i,-1]):
                for j in range(0, position.shape[1]):
                    position.iloc[i,j] = x[j]
            if (best_global[-1] > position.iloc[position['Fitness'].idxmin(),:][-1]):
                best_global = position.iloc[position['Fitness'].idxmin(),:].copy(deep = True)              

        count = count + 1 
        
    print(best_global)    
    return best_global

######################## Part 1 - Usage ####################################

# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

fpa = flower_pollination_algorithm(flowers = 25, min_values = [-5,-5], max_values = [5,5], iterations = 50, gama = 0.1, lamb = 1.5, p = 0.8)
