import numpy as np


def pareto(k, n, beta=1):
    U = np.random.uniform(0, 1, n)
    X = beta*(U**(-1/k)-1) # Had to subtract 1 to correct the graph
    E = (k/(k-1))*beta-1 # Same goes here
    Var = (k / ((k - 1) ** 2 * (k - 2))) * beta ** 2
    return X, E, Var
