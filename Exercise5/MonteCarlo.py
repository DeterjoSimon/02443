import numpy as np


def Crude_Monte_Carlo(n):
    U = np.random.uniform(0, 1, n)
    X = np.exp(U)
    mean = sum(X) / n
    var = sum(X ** 2) / n - mean ** 2
    return X, mean, var
