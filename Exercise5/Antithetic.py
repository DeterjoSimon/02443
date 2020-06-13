import numpy as np


def Antithetic_estimator(n):
    U = np.random.uniform(0, 1, n)
    Yi = (np.exp(U) + np.exp(1 - U)) / 2
    mean = sum(Yi) / n
    var = sum(Yi**2) / n - mean**2
    return Yi, mean, var
