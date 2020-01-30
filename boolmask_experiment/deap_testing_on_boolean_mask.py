"""
The purpose of this file is to explore deap using boolean mask ofn minst datasets

"""
from random import random

from deap.gp import genFull, PrimitiveTree
import numpy as np
from sklearn import datasets
from deap import base, creator, gp, tools


class Bob:

    def __init__(self):
        self.b = np.array([True, True, True, True])

    def __str__(self):
        return self.b


d = datasets.load_digits()


def bool_and(arg_0, arg_1):
    print(f'arg_0 is {type(arg_0)} val is {arg_0} \n arg_1 is {type(arg_1)} val is {arg_1}')
    print(f'return type is {type(np.logical_and(arg_0, arg_1))}')
    print("and")
    return np.logical_and(arg_0, arg_1)


def bool_or(arg_0, arg_1):
    print(f'arg_0 is {type(arg_0)} val is {arg_0} \n arg_1 is {type(arg_1)} val is {arg_1}')
    print(f'return type is {type(np.logical_and(arg_0, arg_1))}')
    print("or")
    return np.logical_or(arg_0, arg_1)


pset = gp.PrimitiveSet('main', 1)
pset.addPrimitive(bool_and, 2)
pset.addPrimitive(bool_or, 2)

#pset.addTerminal(np.array([True, True, True, True]), name='full')
#pset.addTerminal(np.array([False, False, False, False]), name='empty')
# pset.addTerminal(np.array([False, False, True, True]), np.array)
# pset.addTerminal(np.array([True, True, False, False]), np.array)
pset.addEphemeralConstant("t", lambda: np.random.rand(1,4)>0.5)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

expr = genFull(pset, min_=1, max_=3)
tree = PrimitiveTree(expr)
print(tree)

func = gp.compile(tree, pset)
ans = func(np.array([False, True, False, False]))


