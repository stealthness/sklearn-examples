"""
The purpose of this file is to explore deap using boolean mask ofn minst datasets

"""
from random import random

from deap.gp import genFull, PrimitiveTree
import numpy as np
from sklearn import datasets
from deap import base, creator, gp, tools
# importing boolean mask from
from boolmask_experiment.boolean_mask import bool_and, bool_or, bool_not, bool_xor, mask_to_string

LOGGING = True

d = datasets.load_digits()


pset = gp.PrimitiveSet('main', 1)
pset.addPrimitive(bool_and, 2)
pset.addPrimitive(bool_or, 2)
pset.addPrimitive(bool_xor, 2)
pset.addPrimitive(bool_not, 1)

pset.addTerminal(np.array([True, True, True, True]), name='full')
pset.addTerminal(np.array([False, False, False, False]), name='empty')
pset.addTerminal(np.array([False, False, True, True]), name='b0011')
pset.addTerminal(np.array([True, True, False, False]), name='b1100')
pset.addEphemeralConstant("random101", lambda: (np.random.rand(4) > 0.5))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

# genereate full tree ranging from size 1 to 3 depth
expr = genFull(pset, min_=1, max_=3)
tree = PrimitiveTree(expr)

# print out the tree
print(tree)

func = gp.compile(tree, pset)
ans = func(np.array([True, True, True, True]))
print(f'The answer is {mask_to_string(ans)}')


