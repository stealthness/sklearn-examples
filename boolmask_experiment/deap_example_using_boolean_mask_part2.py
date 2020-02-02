"""
The purpose of this file is to explore deap using boolean mask ofn minst datasets

"""
from random import random

from deap.gp import genFull, PrimitiveTree
import numpy as np
from sklearn import datasets, svm, metrics
from deap import base, creator, gp, tools, algorithms
from sklearn.model_selection import train_test_split

from boolmask_experiment.boolean_mask import bool_and, bool_or, bool_not, bool_xor

LOGGING = True


def get_mod_x(data, mask):
    print(mask)
    mod_x = []
    for d in data:
        mod_x.append(d[mask])
    return mod_x

def do_training_and_predicting(data, labels):
    # split the data set into training (80%) and test set(20%)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=109)
    clf = svm.SVC(gamma='auto')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # Import sklearn.metrics to model accuracy
    return metrics.accuracy_score(y_test, y_pred)


d = datasets.load_digits()
x_train, x_test, y_train, _y_test = train_test_split(d.data, d.target, test_size=0.2, random_state=109)

# We added the bool_and, bool_or, ... function to pset
pset = gp.PrimitiveSet('main', 0)
pset.addPrimitive(bool_and, 2)
pset.addPrimitive(bool_or, 2)
pset.addPrimitive(bool_xor, 2)
pset.addPrimitive(bool_not, 1)

# # We add some terminals that are boolean numpy arrays of 64 (8x8)
pset.addTerminal(np.array([True, True, True, False]*16), name='b1110x16')
pset.addTerminal(np.array([False, True, False, False]*16), name='b0100x16')
pset.addTerminal(np.array([False, False, True, True]*16), name='b0011x16')
pset.addTerminal(np.array([True, True, False, False]*16), name='b1100x16')
pset.addEphemeralConstant("random64", lambda: np.random.rand(1, 64) > 0.5)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    result = 1
    if func.sum() > 5:
        mod_x = get_mod_x(d.data, func)
        result = do_training_and_predicting(mod_x, d.target)
    else:
        print(f'fail')
    print(result)
    return result,

toolbox.register("evaluate", eval)
toolbox.register("select", tools.selAutomaticEpsilonLexicase)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)


# test out mod on x_train
p = gp.genFull(pset, min_=1, max_=2)
tree = PrimitiveTree(p)
print(tree)
funct = gp.compile(tree, pset)
t = funct
print(f't is {t}')
print(f'd.data[0] is {d.data[0]}')
print(f'mod_x = {d.data[0][funct]}')
mod_x = get_mod_x(d.data, funct)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, halloffame=hof, verbose=True)
