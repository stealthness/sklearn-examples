"""
The purpose of this file is to explore deap using boolean mask ofn minst datasets

"""
from random import random

from deap.gp import genFull, PrimitiveTree
import numpy as np
from sklearn import datasets
from deap import base, creator, gp, tools

LOGGING = True

class Bob:

    def __init__(self):
        self.b = np.array([True, True, True, True])

    def __str__(self):
        return self.b


d = datasets.load_digits()





pset = gp.PrimitiveSet('main', 1)
pset.addPrimitive(bool_and, 2)
pset.addPrimitive(bool_or, 2)

pset.addTerminal(np.array([True, True, True, True]), name='full')
pset.addTerminal(np.array([False, False, False, False]), name='empty')
# pset.addTerminal(np.array([False, False, True, True]), np.array)
# pset.addTerminal(np.array([True, True, False, False]), np.array)
pset.addEphemeralConstant("t", lambda: np.random.rand(1, 4) > 0.5)

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


