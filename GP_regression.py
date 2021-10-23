import sys
import math
import time
import random
import operator

import numpy as np
from deap import gp, creator, base, tools, algorithms

depth_of_tree = 5
population_size = 2500

def protected_division(left, right):
	if right == 0:
		return 1
	return left / right

f1 = lambda x: protected_division(1, x) + math.sin(x)
f2 = lambda x: (2*x) + (x**2) + 3
f = lambda x: f1(x) if x > 0 else f2(x)

float_range_array = np.arange(-30.0, 30.0, 0.5)
float_range_list = list(float_range_array)

mapping_dictionary = dict()
for x in float_range_array:
	mapping_dictionary[x] = f(x)

# fitness function:
# sum of squared area of function vs given points
def evaluate(individual, reference_function, predicate=(lambda x: True)):
	# Transform the tree expression in a callable function
	func = f1_toolbox.compile(expr=individual)

	# Evaluate the mean squared error between the created function and the given function points
	squared_error = 0.0
	for x in float_range_array:
		if predicate(x):
			squared_error += (func(x) - mapping_dictionary[x])**2
	
	return squared_error, #(1/len(float_range_list))# * squared_error,

def evaluate_f1(individual):
	return evaluate(individual, f1, (lambda val: val > 0))

def evaluate_f2(individual):
	return evaluate(individual, f2, (lambda val: val <= 0))

def evaluate_f(f1_individual, f2_individual):
	return (1/2) * (evaluate_f1(f1_individual) + evaluate_f2(f2_individual))

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(protected_division, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

f1_toolbox = base.Toolbox()
f1_toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
f1_toolbox.register("individual", tools.initIterate, creator.Individual, f1_toolbox.expr)
f1_toolbox.register("population", tools.initRepeat, list, f1_toolbox.individual)
f1_toolbox.register("compile", gp.compile, pset=pset)
f1_toolbox.register("evaluate", evaluate_f1)
f1_toolbox.register("select", tools.selTournament, tournsize=3)
f1_toolbox.register("mate", gp.cxOnePoint)
f1_toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
f1_toolbox.register("mutate", gp.mutUniform, expr=f1_toolbox.expr_mut, pset=pset)
f1_toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_of_tree))
f1_toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_of_tree))

f2_toolbox = base.Toolbox()
f2_toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
f2_toolbox.register("individual", tools.initIterate, creator.Individual, f2_toolbox.expr)
f2_toolbox.register("population", tools.initRepeat, list, f2_toolbox.individual)
f2_toolbox.register("compile", gp.compile, pset=pset)
f2_toolbox.register("evaluate", evaluate_f2)
f2_toolbox.register("select", tools.selTournament, tournsize=3)
f2_toolbox.register("mate", gp.cxOnePoint)
f2_toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
f2_toolbox.register("mutate", gp.mutUniform, expr=f2_toolbox.expr_mut, pset=pset)
f2_toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_of_tree))
f2_toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=depth_of_tree))

if __name__ == '__main__':
	f1_pop = f1_toolbox.population(n=population_size)
	f2_pop = f2_toolbox.population(n=population_size)

	f1_best = tools.HallOfFame(1)
	f2_best = tools.HallOfFame(1)
	
	f1_pop, log = algorithms.eaSimple(f1_pop, f1_toolbox, 0.5, 0.1, 40, halloffame=f1_best, verbose=False)
	f2_pop = algorithms.eaSimple(f2_pop, f2_toolbox, 0.5, 0.1, 40, halloffame=f2_best, verbose=False)

	print("\n------------------------------------------------------------")
	print('''
{} if x > 0
{} if x ≤ 0'''.format(f1_best[0], f2_best[0]))
	print("fitness = {}".format(evaluate_f1(f1_best[0]))) # this is the line I will need to change: some way of evaluating the whole function
	print("Note that a fitness of 0 means that there is zero squared error between the found function and the function we were searching for.")
	print("------------------------------------------------------------\n")