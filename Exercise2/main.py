from Exercise2 import geometric
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def histogram(x, y, n_bins=10):
    plt.hist([x, y], n_bins, alpha=0.5, label=["Simulated", "In-built"])
    plt.legend()
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.show()


def direct_crude(U, p):
    X = []
    for k in range(len(U)):
        for i in range(1, len(p)):
            if U[k] <= np.sum(p[:i]):
                X.append(i)
                break
    return np.array(X)


def rejection(U, p):
    X = []
    p = np.array(p)
    qj = np.full(len(p), sum(p)/len(p))
    c = max(p/qj)
    while len(X) < len(U):

        I = np.floor(len(p)*np.random.uniform(size=1)) + 1
        u2 = np.random.uniform(size=1)
        if u2 <= p[I]/c:
            X.append(I)


if __name__ == '__main__':
    # generate 10,000 uniform random variables
    U = np.random.random_sample(10000)  # 0 og 1

    # simulate 10 000
    p = 0.3
    A = geometric.geometric(U, p)
    # we compare this geometric distribution with a built-in one
    B = stats.geom.rvs(p, size=10000)
    # histogram(A, B)

    p = [7 / 48, 5 / 48, 1 / 8, 1 / 16, 1 / 4, 5 / 16]

    a = direct_crude(U, p)
    print(a)
    print(sum(a[a == 1]))

