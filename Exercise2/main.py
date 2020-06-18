from Exercise2 import geometric
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random as rand
import math


def histogram(x, y, label1, label2, n_bins=10):
    plt.hist([x, y], bins=n_bins, alpha=0.5, label=[label1, label2])
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


def rejection(p):
    X = []
    p = np.array(p)
    qj = np.full(len(p), sum(p)/len(p))
    c = max(p/qj)
    while len(X) < len(U):

        I = math.floor(len(p)*rand.uniform(0, 1)) + 1
        u2 = np.random.uniform(size=1)
        if u2 <= p[I-1]/c:
            X.append(I)
    return X


def alias(p):
    k = len(p)
    L = [i for i in range(k)]
    F = [i*k for i in p]
    G = [i for i in range(len(F)) if F[i] >= 1]
    S = [i for i in range(len(F)) if F[i] <= 1]

    while len(S) != 0:
        i, j = G[0], S[0]
        L[j-1] = i
        F[i-1] = F[i-1] - (1 - F[j-1])

        if F[i-1] < 1:
            G = G[1:]
            S.append(i)
        S = S[1:]

    X = []
    while len(X) < 10000:
        I = math.floor(k * np.random.uniform(0, 1)) + 1
        u2 = np.random.uniform(0, 1)
        if u2 < F[I-1]:
            X.append(I)
        else:
            X.append(L[I-1])
    return X


if __name__ == '__main__':
    # generate 10,000 uniform random variables
    U = np.random.random_sample(10000)  # 0 og 1

    # simulate 10 000
    p = 0.3
    A = geometric.geometric(U, p)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].hist(geometric.geometric(U, p=0.1), alpha=0.7, density=True, stacked=True)
    axes[0].set_title("p = 0.1")
    axes[1].hist(geometric.geometric(U, p=0.5), alpha=0.7, density=True, stacked=True)
    axes[1].set_title("p = 0.5")
    axes[2].hist(geometric.geometric(U, p=0.7), alpha=0.7, density=True, stacked=True)
    axes[2].set_title("p = 0.7")
    plt.show()
    # we compare this geometric distribution with a built-in one
    B = stats.geom.rvs(p, size=10000)
    histogram(A, B, "simulation", "built-in")

    print("--- Chi-square test ---")

    t_test, p_value = stats.chisquare(np.histogram(A, bins=10)[0], np.histogram(B, bins=10)[0])
    print("We get a test statistic of {0}".format(t_test))

    p = [7 / 48, 5 / 48, 1 / 8, 1 / 16, 1 / 4, 5 / 16]
    expected = stats.rv_discrete(values=(np.arange(1, 7), p)).rvs(size=10000)
    a = direct_crude(U, p)
    histogram(expected, a, "Expected", "Direct Crude")

    print("--- Chi-square test ---")

    t_test, p_value = stats.chisquare(np.histogram(expected, bins=6, density=False)[0], np.histogram(a, bins=6, density=False)[0])
    print("We get a test statistic of {0}".format(t_test))

    b = rejection(p)
    histogram(expected, b, "Expected", "Rejection")

    print("--- Chi-square test ---")

    t_test, p_value = stats.chisquare(np.histogram(expected, bins=6, density=False)[0],
                                      np.histogram(a, bins=6, density=False)[0])
    print("We get a test statistic of {0}".format(t_test))

    c = alias(p)
    histogram(expected, c, "Expected", "Alias")
    print("--- Chi-square test ---")

    t_test, p_value = stats.chisquare(np.histogram(expected, bins=6, density=False)[0],
                                      np.histogram(a, bins=6, density=False)[0])
    print("We get a test statistic of {0}".format(t_test))
