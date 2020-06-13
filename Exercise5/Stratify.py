import numpy as np


def stratify_estimator(n):
    U = np.random.uniform(1, 0, n)
    W = 0
    for i in range(10):
        W += np.exp(i/10 + U/10)
    W /= 10
    meanW = sum(W)/n
    varW = sum(W**2)/n - meanW**2
    return W, meanW, varW