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
from scipy.special import gamma

# Function
def target_function():
    return

# Function: Initialize Variables
def initial_position_(flowers = 3, min_values = [-5,-5], max_values = [5,5]):
    position = pd.DataFrame(np.zeros((flowers, len(min_values))))
    position['Fitness'] = 0.0
    for i in range(0, flowers):
        for j in range(0, len(min_values)):
             position.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        position.iloc[i,-1] = target_function(position.iloc[i,0:position.shape[1]-1])
    return position

# Function: Initialize Variables
def initial_position(flowers = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((flowers, len(min_values)+1))
    for i in range(0, flowers):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

#Function Levy Distribution
def levy_flight(beta = 1.5):
    beta    = beta  
    r1      = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    r2      = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    sig_num = gamma(1+beta)*np.sin((np.pi*beta)/2.0)
    sig_den = gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma   = (sig_num/sig_den)**(1/beta)
    levy    = (0.01*r1*sigma)/(abs(r2)**(1/beta))
    return levy

# Function: Global Pollination
def pollination_global_(position, best_global, flower = 0, gama = 0.5, lamb = 1.4, min_values = [-5,-5], max_values = [5,5]):
    x = best_global.copy(deep = True)
    for j in range(0, len(min_values)):
        x[j] = position.iloc[flower, j]  + gama*levy_flight(lamb)*(position.iloc[flower, j] - best_global[j])
        if (x[j] > max_values[j]):
            x[j] = max_values[j]
        elif(x[j] < min_values[j]):
            x[j] = min_values[j]
    x[-1]  = target_function(x[0:len(min_values)])
    return x

# Function: Global Pollination
def pollination_global(position, best_global, flower = 0, gama = 0.5, lamb = 1.4, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    x = np.copy(best_global)
    for j in range(0, len(min_values)):
        x[j] = np.clip((position[flower, j]  + gama*levy_flight(lamb)*(position[flower, j] - best_global[j])),min_values[j],max_values[j])
    x[-1]  = target_function(x[0:len(min_values)])
    return x

# Function: Local Pollination
def pollination_local_(position, best_global, flower = 0, nb_flower_1 = 0, nb_flower_2 = 1, min_values = [-5,-5], max_values = [5,5]):
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

# Function: Local Pollination
def pollination_local(position, best_global, flower = 0, nb_flower_1 = 0, nb_flower_2 = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    x = np.copy(best_global)
    for j in range(0, len(min_values)):
        r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        x[j] = np.clip((position[flower, j]  + r*(position[nb_flower_1, j] - position[nb_flower_2, j])),min_values[j],max_values[j])
    x[-1]  = target_function(x[0:len(min_values)])
    return x

# FPA Function.
def flower_pollination_algorithm_(flowers = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, gama = 0.5, lamb = 1.4, p = 0.8):    
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

# FPA Function.
def flower_pollination_algorithm(flowers = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, gama = 0.5, lamb = 1.4, p = 0.8, target_function = target_function):    
    count    = 0
    position = initial_position(flowers = flowers, min_values = min_values, max_values = max_values, target_function = target_function)
    best_global = np.copy(position[position[:,-1].argsort()][0,:])
    x = np.copy(best_global)   
    while (count <= iterations):
        print("Iteration = ", count, " f(x) = ", best_global[-1])
        for i in range(0, position.shape[0]):
            nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
            nb_flower_2 = int(np.random.randint(position.shape[0], size = 1))
            while nb_flower_1 == nb_flower_2:
                nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
            r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (r < p):
                x = pollination_global(position, best_global, flower = i, gama = gama, lamb = lamb, min_values = min_values, max_values = max_values, target_function = target_function)
            else:
                x = pollination_local(position, best_global, flower = i, nb_flower_1 = nb_flower_1, nb_flower_2 = nb_flower_2, min_values = min_values, max_values = max_values, target_function = target_function)
            if (x[-1] <= position[i,-1]):
                for j in range(0, position.shape[1]):
                    position[i,j] = x[j]
            value = np.copy(position[position[:,-1].argsort()][0,:])
            if (best_global[-1] > value[-1]):
                best_global = np.copy(value) 
        count = count + 1       
    print(best_global)    
    return best_global

######################## Part 1 - Usage ####################################

# Function to be Minimized (Six Hump Camel Back). Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

fpa = flower_pollination_algorithm(flowers = 25, min_values = [-5,-5], max_values = [5,5], iterations = 500, gama = 0.1, lamb = 1.5, p = 0.8, target_function = six_hump_camel_back)

# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

fpa = flower_pollination_algorithm(flowers = 175, min_values = [-5,-5], max_values = [5,5], iterations = 1000, gama = 0.1, lamb = 1.5, p = 0.8, target_function = rosenbrocks_valley)
