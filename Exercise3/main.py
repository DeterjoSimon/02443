from Exercise3 import exponential, normal, funcPareto
import scipy.stats
from scipy.stats import expon, t, norm, pareto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def comparison(hist, ppf, pdf, title):

    plt.figure()
    plt.hist(hist, bins=10, density=True, color='g', alpha=0.5, stacked=True)
    plt.plot(ppf, pdf, 'r--')
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Density')
    plt.xlim(-10, 50)
    plt.show()


if __name__ == "__main__":

    # histogram 1 for exponential
    U = np.random.uniform(0.0, 0.1, 10000)
    ppf = np.linspace(expon.ppf(0), expon.ppf(0.99999), 100)
    pdf = expon.pdf(ppf)
    # comparison(exponential.exponential(U, 1), ppf, pdf, "Comparison histogram for exponential distribution")

    # # histogram 2 for normal
    ppf = np.linspace(-5, 5, 100)
    pdf = norm.pdf(ppf, 0.0, 1.0)
    h = normal.normal(10000)
    # comparison(h, ppf, pdf, "Comparison histogram for normal random variables using Box-Muller")
    m, up, low = mean_confidence_interval(h)
    # plt.figure()
    # CI = np.array([low, up])
    # plt.plot(CI[0], 'r-', label='lower')
    # plt.plot(CI[1], 'c-', label='higher')
    # plt.show()

    # histogram 3 for Pareto
    k = [2.05, 2.5, 3, 4]
    # mean_list = []
    # var_list = []
    # dic = {"k:2.05": [], "k:2.5": [], "k:3": [], "k:4": []}
    # for i in k:
    #     U = np.random.uniform(0.0, 0.1, 10000)
    #     par1, E, Var = funcPareto.pareto(U, i)
    #     ppf = np.linspace(pareto.ppf(0.01, i), pareto.ppf(0.99, i), 100)
    #     pdf = pareto.pdf(ppf, i)
    #     comparison(par1, ppf-1, pdf, "Pareto comparison for k = {0}".format(i))
    #     dic["k:{0}".format(i)] = [E, Var, np.mean(par1), np.var(par1)]
    # df = pd.DataFrame(dic, index=["Mean", "Variance", "Mean_analytical", "Variance_analytical"])
    # print(df)

    # Normal 100 CI's
    upper_l = []
    lower_l = []
    for i in range(100):
        X = normal.normal(10)
        upper_l.append(np.mean(X) + 1.96*(np.std(X))/math.sqrt(10))
        lower_l.append(np.mean(X) - 1.96*(np.std(X))/math.sqrt(10))
    plt.figure()
    plt.hist(upper_l, color='g', label='Upper Limit', alpha=0.5)
    plt.hist(lower_l, color='r', label='Lower Limit', alpha=0.5)
    plt.title('Confidence Interval histogram')
    plt.legend()
    plt.show()