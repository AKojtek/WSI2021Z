from matplotlib import pyplot as plt
import numpy as np
import random
import math
import copy

population = [
        [-24, -28, -67, -5, 28, -18, -32, -11, 32, -62],
        [-73, 31, 60, -23, 76, -100, -46, 38, 99, -52],
        [-50, -3, -58, 11, -92, -32, -41, -19, 98, -100],
        [-2, -83, 86, 85, -34, -75, 52, 44, -11, -16],
        [6, 21, -16, -17, 52, 47, 0, -19, -60, 49],
        [86, -97, -58, 3, 27, 5, -79, -97, -30, -61],
        [-93, 86, -5, -71, 76, -57, -93, -52, 15, 4],
        [75, -8, -69, -16, -39, 56, -100, -40, -32, -4],
        [-19, 60, 88, 32, 61, -16, 21, -59, -17, -93],
        [37, 52, 12, 43, -100, -21, 18, -70, -77, -50],
        [-65, 45, -12, -69, 28, -87, -89, -73, -90, 24],
        [-14, 81, -35, -20, 92, -98, 24, -35, 94, 66],
        [70, -69, 93, -99, 53, 12, -64, 2, 10, -50],
        [-41, 10, -44, 97, -16, -30, 72, 52, -72, 16],
        [-48, 68, 0, -35, 53, -65, -54, -54, 35, 6],
        [-40, -15, -26, 60, -74, 20, 76, 97, -64, -39],
        [10, -47, -52, -1, 66, 77, 4, -34, -60, 45],
        [71, -40, 78, -73, 6, -42, -92, 29, -70, -69],
        [-95, 9, -23, -6, -36, -16, -7, 13, -67, -21],
        [78, 68, 73, -50, -62, -28, -62, 93, -51, -6]
    ]

Dim = 10
tests_number = 5


def generate_population(number, dimen, seed):
    random.seed(seed)
    population = []
    for _ in range(number):
        speciman = []
        for _ in range(dimen):
            speciman.append(random.randrange(-100,100))
        population.append(speciman)
    return population

def average_pop(population, dimen):
    average = [0]*dimen
    for each in population:
        for i in range(dimen):
            average[i] += each[i]
    return [x / len(population) for x in average]

def target_function(speciman, d):
    norm_sq = np.linalg.norm(speciman)**2
    res = ((norm_sq - d)**2)**(1/8)
    res += (norm_sq/2 + sum(speciman))/d
    res += 1/2
    return res

def sphere_function(speciman, size):
    total = 0
    for each in speciman:
        total += each*each
    return total

def selfadaptation(mutation_str, population, descendants_no, dimen, target_fun, seed, max_iter):
    rng = np.random.default_rng(seed)
    tau = 1 / math.sqrt(dimen)
    pop_size = len(population)
    t = 0
    best_specimen = 0
    best_specimens = []
    average_specimen = []
    while(t < max_iter):
        descendants = []
        average = average_pop(population, dimen)
        average_specimen.append(target_fun(average,dimen))
        for _ in range(descendants_no):
            ksi = tau * rng.normal(0,1)
            z =  [rng.normal(0,1) for _ in range(dimen)]
            mutation_t = mutation_str * math.exp(ksi)
            for i in range(dimen):
                z[i] *= mutation_t
            descendant = []
            for i in range(dimen):
                descendant.append(average[i]+z[i])
            fun_result = abs(target_fun(descendant, dimen))
            descendants.append((mutation_t, fun_result, descendant))
        descendants.sort(key=lambda a: a[1])
        if t == 0:
            best_specimen = descendants[0][1]
        else:
            if best_specimen > descendants[0][1]:
                best_specimen = descendants[0][1]
        best_specimens.append(best_specimen)
        new_str = 0
        for i in range(pop_size):
            new_str += descendants[i][0]
            population[i] = descendants[i][2]
        mutation_str = new_str / pop_size
        t += 1
    return best_specimens, average_specimen


def gauss_logarithm(mutation_str, population, descendants_no, dimen, target_fun, seed, max_iter):
    rng = np.random.default_rng(seed)
    tau = 1/ math.sqrt(dimen)
    t = 0
    pop_size = len(population)
    best_specimen = 0
    best_specimens = []
    average_specimen = []
    while(t < max_iter):
        descendants = []
        average = average_pop(population, dimen)
        average_specimen.append(target_fun(average,dimen))
        for _ in range(descendants_no):
            z =  [rng.normal(0,1) for _ in range(dimen)]
            for i in range(dimen):
                z[i] *= mutation_str
            descendant = []
            for i in range(dimen):
                descendant.append(average[i]+z[i])
            fun_result = abs(target_fun(descendant, dimen))
            descendants.append((mutation_str, fun_result, descendant))
        descendants.sort(key=lambda a: a[1])
        if t == 0:
            best_specimen = descendants[0][1]
        else:
            if best_specimen > descendants[0][1]:
                best_specimen = descendants[0][1]
        best_specimens.append(best_specimen)
        for i in range(pop_size):
            population[i] = descendants[i][2]
        mutation_str *= math.exp(tau * rng.normal(0,1))
        t += 1
    return best_specimens, average_specimen

def experiment_average(sigma, population, lambd, D, target_fun, seed, iterations_no, method, test_numbers):
    keys_ave = method(sigma,population,lambd,D,target_fun,seed,iterations_no)[1]
    # for i in range(1,test_numbers):
    #     data = method(sigma,population,lambd,D,target_fun,seed+i,iterations_no)
    #     for j in range(len(population)):
    #         keys_ave[j] = data[1][j]
    # for i in range(len(population)):
    #     keys_ave[i] /= test_numbers
    return keys_ave

def experiment_best(sigma, population, lambd, D, target_fun, seed, iterations_no, method, test_numbers):
    keys_best = method(sigma,population,lambd,D,target_fun,seed,iterations_no)[0]
    # for i in range(1,test_numbers):
    #     data = method(sigma,population,lambd,D,target_fun,seed+i,iterations_no)
    #     for j in range(len(population)):
    #         keys_best[j] = data[0][j]
    # for i in range(len(population)):
    #     keys_best[i] /= test_numbers
    return keys_best

# plt.clf()
# ave_data = experiment_average(0.01, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="0.01")
# ave_data = experiment_average(0.1, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="0.1")
# ave_data = experiment_average(0.5, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="0.5")
# ave_data = experiment_average(1, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="1")
# ave_data = experiment_average(5, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="2")
# ave_data = experiment_average(10, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="4")
# title = f'Srednia, zmienna sigma, SA, Funkcja sferyczna'
# plt.title(title)
# plt.xlabel('Ilość iteracji')
# plt.ylabel('Wartość funkcji')
# plt.legend()
# plt.savefig('srednia5testow.png')

# plt.clf()
# ave_data = experiment_average(0.01, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="0.01")
# ave_data = experiment_average(0.1, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="0.1")
# ave_data = experiment_average(0.5, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="0.5")
# ave_data = experiment_average(1, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="1")
# ave_data = experiment_average(5, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="2")
# ave_data = experiment_average(10, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
# plt.plot([*range(0,100)], ave_data, '-', label="4")
# title = f'Srednia, zmienna sigma, SA, Funkcja z zajec'
# plt.title(title)
# plt.xlabel('Ilość iteracji')
# plt.ylabel('Wartość funkcji')
# plt.legend()
# plt.savefig('srednia5testow2.png')

plt.clf()
ave_data = experiment_average(0.01, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.01")
ave_data = experiment_average(0.1, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.1")
ave_data = experiment_average(0.5, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.5")
ave_data = experiment_average(1, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="1")
ave_data = experiment_average(5, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="2")
ave_data = experiment_average(10, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="4")
title = f'Srednia, zmienna sigma, SA, Funkcja sferyczna'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test1.png')

plt.clf()
ave_data = experiment_average(0.01, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.01")
ave_data = experiment_average(0.1, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.1")
ave_data = experiment_average(0.5, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.5")
ave_data = experiment_average(1, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="1")
ave_data = experiment_average(5, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="2")
ave_data = experiment_average(10, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="4")
title = f'Srednia, zmienna sigma, SA, Funkcja z zajec'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test2.png')

plt.clf()
ave_data = experiment_average(0.01, population, 40, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.01")
ave_data = experiment_average(0.1, population, 40, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.1")
ave_data = experiment_average(0.5, population, 40, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.5")
ave_data = experiment_average(1, population, 40, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="1")
ave_data = experiment_average(5, population, 40, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="2")
ave_data = experiment_average(10, population, 40, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="4")
title = f'Srednia, zmienna sigma, GL, Funkcja sferyczna'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test3.png')

plt.clf()
ave_data = experiment_average(0.01, population, 40, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.01")
ave_data = experiment_average(0.1, population, 40, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.1")
ave_data = experiment_average(0.5, population, 40, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="0.5")
ave_data = experiment_average(1, population, 40, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="1")
ave_data = experiment_average(5, population, 40, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="2")
ave_data = experiment_average(10, population, 40, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="4")
title = f'Srednia, zmienna sigma, GL, Funkcja z zajec'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test4.png')


# zmienna lambda

plt.clf()
ave_data = experiment_average(4, population, 20, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="20")
ave_data = experiment_average(4, population, 25, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="25")
ave_data = experiment_average(4, population, 30, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="30")
ave_data = experiment_average(4, population, 35, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="35")
ave_data = experiment_average(4, population, 40, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="40")
ave_data = experiment_average(4, population, 50, Dim, sphere_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="45")
title = f'Srednia, zmienna lambda, SA, Funkcja sferyczna'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test5.png')

plt.clf()
ave_data = experiment_average(4, population, 20, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="20")
ave_data = experiment_average(4, population, 25, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="25")
ave_data = experiment_average(4, population, 30, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="30")
ave_data = experiment_average(4, population, 35, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="35")
ave_data = experiment_average(4, population, 40, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="40")
ave_data = experiment_average(4, population, 45, Dim, target_function, 1, 100, selfadaptation, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="45")
title = f'Srednia, zmienna lamda, SA, Funkcja z zajec'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test6.png')

plt.clf()
ave_data = experiment_average(4, population, 20, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="20")
ave_data = experiment_average(4, population, 25, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="25")
ave_data = experiment_average(4, population, 30, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="30")
ave_data = experiment_average(4, population, 35, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="35")
ave_data = experiment_average(4, population, 40, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="40")
ave_data = experiment_average(4, population, 45, Dim, sphere_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="45")
title = f'Srednia, zmienna lambda, GL, Funkcja sferyczna'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test7.png')

plt.clf()
ave_data = experiment_average(4, population, 20, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="20")
ave_data = experiment_average(4, population, 25, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="25")
ave_data = experiment_average(4, population, 30, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="30")
ave_data = experiment_average(4, population, 35, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="35")
ave_data = experiment_average(4, population, 40, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="40")
ave_data = experiment_average(4, population, 45, Dim, target_function, 1, 100, gauss_logarithm, tests_number)
plt.plot([*range(0,100)], ave_data, '-', label="45")
title = f'Srednia, zmienna lambda, GL, Funkcja z zajec'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('test8.png')

plt.clf()
ave_data = experiment_average(0.01, population, 40, Dim, target_function, 1, 1000, gauss_logarithm, tests_number)
plt.plot([*range(0,1000)], ave_data, '-', label="0.01")
ave_data = experiment_average(0.1, population, 40, Dim, target_function, 1, 1000, gauss_logarithm, tests_number)
plt.plot([*range(0,1000)], ave_data, '-', label="0.1")
ave_data = experiment_average(0.5, population, 40, Dim, target_function, 1, 1000, gauss_logarithm, tests_number)
plt.plot([*range(0,1000)], ave_data, '-', label="0.5")
ave_data = experiment_average(1, population, 40, Dim, target_function, 1, 1000, gauss_logarithm, tests_number)
plt.plot([*range(0,1000)], ave_data, '-', label="1")
ave_data = experiment_average(2, population, 40, Dim, target_function, 1, 1000, gauss_logarithm, tests_number)
plt.plot([*range(0,1000)], ave_data, '-', label="2")
ave_data = experiment_average(4, population, 40, Dim, target_function, 1, 1000, gauss_logarithm, tests_number)
plt.plot([*range(0,1000)], ave_data, '-', label="4")
title = f'1000 iteracji'
plt.title(title)
plt.xlabel('Ilość iteracji')
plt.ylabel('Wartość funkcji')
plt.legend()
plt.savefig('1000_iteracji.png')

