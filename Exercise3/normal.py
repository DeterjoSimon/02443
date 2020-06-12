import numpy as np


def normal(n):
    result = []
    for i in range(n):
        U1 = np.random.uniform(0, 1)
        U2 = np.random.uniform(0, 1)
        Rsquared = -2 * np.log(U1)
        Theta = 2*np.pi*U2
        X = np.sqrt(Rsquared)*np.cos(Theta)
        Y = np.sqrt(Rsquared)*np.sin(Theta)
        result = np.append(result, np.array([X]))
    return result
