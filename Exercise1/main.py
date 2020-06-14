from Exercise1 import LCG as lcg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import math


def histogram():
    n, bins, patches = plt.hist(x=U, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


def chi_squared(X, n, n_classes=10):
    expected = np.full(10, n/n_classes)
    obs, bins = np.histogram(X, n_classes)

    test_stat = sum(((obs-expected)**2)/expected)
    return test_stat


def ks(X, n):
    F = sorted(X)
    Fn = np.linspace(0, 1, n)
    D = np.max(abs(Fn-F))
    return D


if __name__ == '__main__':
    M = 2**38 - 45
    U = lcg.lcg(13, 911, M, 3)
    print("--- Chi-squared test---")
    print(chi_squared(U, 10000), " : Value of the test statistic")
    print("[" + str(chi2.ppf(0.025, 8)) + " ; " + str(chi2.ppf(0.975, 8)) + "]")

    print("--- Kolmogorov-Smirnov test")
    print("The Kolmogorov–Smirnov statistic is: ", ks(U, 10000))
