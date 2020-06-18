from Exercise1 import LCG as lcg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import math
import scipy.stats
from collections import Counter


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
    expected = np.full(10, n / n_classes)
    obs, bins = np.histogram(X, n_classes)

    test_stat = sum(((obs - expected) ** 2) / expected)
    return test_stat


def ks(X, n):
    F = sorted(X)
    Fn = np.linspace(0, 1, n)
    D = np.max(abs(Fn - F))
    return D


def run_test_I(X, n):
    n1 = len(X[X > np.median(X)])
    n2 = len(X[X < np.median(X)])
    med = np.median(X)
    Ra = 0
    Rb = 0
    runs = 0
    crit = None
    for i in X:
        if i < med and crit != 'below':
            Ra += 1
            runs += 1
            crit = 'below'
        elif i > med and crit != 'above':
            Rb += 1
            runs += 1
            crit = 'above'
    mu = 2*n1*n2/(n1+n2)+1
    sigma = 2*n1*n2*(2*n1*n2-n1-n2)/((n1+n2)**2*(n1+n2-1))
    T = Ra + Rb
    return T, mu, sigma


def run_test_II(X, n):
    k = 1
    runs = []
    for i in range(len(X) - 1):
        if X[i] <= X[i + 1]:
            k += 1
        else:
            runs.append(k)
            k = 1
    runs.append(k)
    runs = np.array(runs)
    R = []
    for i in range(1, 7):
        R.append(len(runs[runs == i]))
    R = np.array(R)
    B = np.array([1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 840])
    A = np.array(
        [[4529.4, 9044.9, 13568, 18091, 22615, 27892],
         [9044.9, 18097, 27139, 36187, 45234, 55789],
         [13568, 27139, 40721, 54281, 67852, 83685],
         [18091, 36187, 54281, 72414, 90470, 111580],
         [22615, 45234, 67852, 90470, 113262, 139476],
         [27892, 55789, 83685, 111580, 139476, 172860]])
    temp = R - n * B
    Z = (1 / (n - 6)) * np.dot(temp.T, np.dot(A, temp))
    return Z


def run_test_III(X, n):
    runs = np.zeros(7)
    up = 1
    down = 1
    for i in range(1, len(X) - 1):
        if X[i - 1] < X[i]:
            up += 1
            if down != 0:
                runs[down] += 1
            down = 0
        if X[i - 1] > X[i]:
            down += 1
            if up != 0:
                runs[up] += 1
            up = 0
    runs[max([up, down])] += 1
    Z = (sum(runs) - (2*n - 1)/3)/math.sqrt((16*n - 29)/90)
    return runs[1:], Z


if __name__ == '__main__':
    M = 2 ** 19 - 1
    U = lcg.lcg(47, 45, M, 3)
    print("--- Chi-squared test---")
    print(chi_squared(U, 10000), " : Value of the test statistic")
    print("[" + str(chi2.ppf(0.025, 8)) + " ; " + str(chi2.ppf(0.975, 8)) + "]")

    print("\n--- Kolmogorov-Smirnov test ---")
    D = ks(U, 10000)
    print("The Kolmogorov–Smirnov statistic is: ", D)
    print("Test Statistic is: ", (math.sqrt(10000) + 0.12 + 0.11 / math.sqrt(10000)) * D)

    print("\n--- Run Test I ---")
    print("test statistic: ", run_test_I(U, 10000)[0])
    T, mu, sigma = run_test_I(U, 10000)
    print("[", scipy.stats.norm.ppf(0.025, loc=mu, scale=sigma), ";", scipy.stats.norm.ppf(0.975, loc=mu, scale=sigma),"]")

    print("\n--- Run Test II ---")
    print("test statistic: ", run_test_II(U, 10000))
    print("[", scipy.stats.chi2.ppf(0.025, 6), ";", scipy.stats.chi2.ppf(0.975, 6), "]")

    print("\n--- Run Test III ---")
    print(run_test_III(U, 10000)[0])
    for i in range(1, 6):
        print("Expected for length {0}: {1}".format(i, (2 * ((i ** 2 + 3 * i + 1) * 10000 -
                                                             (i ** 3 + 3 * i ** 2 - i - 4))) / math.factorial(i + 3)))
    print("Test statistic: ", run_test_III(U, 10000)[1])
    print("[", scipy.stats.norm.ppf(0.025), ";", scipy.stats.norm.ppf(0.975), "]")

    # plt.scatter(U, np.linspace(0, 1, len(U)), marker='x', alpha=0.5)
    # plt.title("U[i+1] vs. U[i]")
    # plt.xlabel("U")
    # plt.show()
    # print(U)
    plt.scatter(lcg.lcg(5, 1, 16, 3), np.linspace(0, 1, len(lcg.lcg(5, 1, 16, 3))), marker='x', alpha=0.5)
    plt.title("U[i+1] vs. U[i]")
    plt.show()
    print("------------------------------------------------------")
    print("--- For In-Built function ---")
    U = np.random.rand(10000)
    print("--- Chi-squared test---")
    print(chi_squared(U, 10000), " : Value of the test statistic")
    print("[" + str(chi2.ppf(0.025, 8)) + " ; " + str(chi2.ppf(0.975, 8)) + "]")

    print("--- Kolmogorov-Smirnov test ---")
    D = ks(U, 10000)
    print("The Kolmogorov–Smirnov statistic is: ", D)
    print("Test Statistic is: ", (math.sqrt(10000) + 0.12 + 0.11 / math.sqrt(10000)) * D)

    print("\n--- Run Test I ---")
    print("test statistic: ", run_test_I(U, 10000)[0])
    T, mu, sigma = run_test_I(U, 10000)
    print("[", scipy.stats.norm.ppf(0.025, loc=mu, scale=sigma), ";", scipy.stats.norm.ppf(0.975, loc=mu, scale=sigma),
          "]")

    print("\n--- Run Test II ---")
    print("test statistic: ", run_test_II(U, 10000))
    print("[", scipy.stats.chi2.ppf(0.025, 6), ";", scipy.stats.chi2.ppf(0.975, 6), "]")

    print("\n--- Run Test III ---")
    print(run_test_III(U, 10000)[0])
    for i in range(1, 7):
        print("Expected for length {0}: {1}".format(i, (2 * ((i ** 2 + 3 * i + 1) * 10000 -
                                                             (i ** 3 + 3 * i ** 2 - i - 4))) / math.factorial(i + 3)))
    print("Test statistic: ", run_test_III(U, 10000)[1])
    print("[", scipy.stats.norm.ppf(0.025), ";", scipy.stats.norm.ppf(0.975), "]")