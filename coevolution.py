import sys
import math
import time
import random
import operator

import numpy as np
from deap import gp, creator, base, tools, algorithms

print("\nNote that this takes a while to run with the current parameters (~30 seonds on my computer). If you want a faster run, feel free to drop the population_size parameter at the top of the code. This will probably give less accurate functions (with higher fitness).")

depth_of_tree = 7
population_size = 2700

def function_to_find(x):
	if x <= 0:
		return (2*x)+(x*x)+3.0
	else:
		return (1/x) + math.sin(x) # might need to change to protected division

# generates all the x values we need to assess our function at
float_range_array = np.arange(-5.0, 5.0, 0.25)
float_range_list = list(float_range_array)

mapping_dictionary = dict()
for x in float_range_array:
	mapping_dictionary[x] = function_to_find(x)

def if_function(x, constant, individual1, individual2):
	if x <= constant:
		return individual1
	else:
		return individual2

def protected_division(left, right):
	if right == 0:
		return 1
	return left / right

# fitness function:
# sum of squared area of function vs given points
def evaluate_symbol_regression(individual):
	# Transform the tree expression in a callable function
	func = toolbox.compile(expr=individual)

	#print(individual)
	# Evaluate the mean squared error between the created function and the given function points
	squared_error = 0
	for x in float_range_array:
		if ("sin(protected_division" in str(individual)) and x == 0.0:
			continue
		mapping_dictionary[x]
		squared_error += (func(x) - mapping_dictionary[x])**2
	
	return (1 / len(float_range_list)) * squared_error,

if __name__ == '__main__':
	program_t0 = time.time()

	pset = gp.PrimitiveSet("MAIN", 1)
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(operator.neg, 1)
	pset.addPrimitive(protected_division, 2)
	pset.addPrimitive(if_function, 4)
	pset.addPrimitive(math.sin, 1)
	pset.addPrimitive(math.cos, 1)
	pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
	pset.renameArguments(ARG0="x")
	pset.renameArguments(ARG1="y")

	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

	toolbox = base.Toolbox()
	toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("compile", gp.compile, pset=pset)

	toolbox.register("evaluate", evaluate_symbol_regression)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
	toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
	
	toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_of_tree))
	toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_of_tree))

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", np.mean)
	mstats.register("std", np.std)
	mstats.register("min", np.min)
	mstats.register("max", np.max)

	pop = toolbox.population(n=population_size)
	hof = tools.HallOfFame(1)

	pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=False)
	
	print("\n------------------------------------------------------------")
	print("function = {}\n".format(hof[0]))
	print("fitness = {}".format(evaluate_symbol_regression(hof[0])))
	print("------------------------------------------------------------\n")

	print(f"program runtime = {time.time() - program_t0}")