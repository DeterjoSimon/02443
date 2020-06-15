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


def std_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    std, se = np.std(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return std, std-h, std+h


def comparison(hist, ppf, pdf, title):

    plt.figure()
    plt.hist(hist, bins=10, density=True, color='g', alpha=0.5, stacked=True, label="Pareto")
    plt.plot(ppf, pdf, 'r--', label="In-Built Pareto")
    plt.title(title)
    plt.legend()
    plt.xlabel('Classes')
    plt.ylabel('Density')
    plt.xlim(0, 10)
    plt.show()


if __name__ == "__main__":

    # histogram 1 for exponential
    U = np.random.uniform(0.0, 0.1, 10000)
    ppf = np.linspace(expon.ppf(0), expon.ppf(0.9), 100)
    pdf = expon.pdf(ppf)
    # comparison(exponential.exponential(U, 1), ppf, pdf, "Comparison histogram for exponential distribution")

    # # histogram 2 for normal
    ppf = np.linspace(-5, 5, 100)
    pdf = norm.pdf(ppf, 0.0, 1.0)
    h = normal.normal(10000)
    # comparison(h, ppf, pdf, "Comparison histogram for normal random variables using Box-Muller")
    # # Bootstrap method to build confidence interval
    # mean = []
    # var = []
    # for i in range(100):
    #     x = normal.normal(10)
    #     m, low1, up1 = mean_confidence_interval(x)
    #     std, low2, up2 = std_confidence_interval(x)
    #     mean.append([low1, up1])
    #     var.append([low2, up2])
    # mean, var = np.array(mean), np.array(var)
    # # plot the upper and lower limit
    # plt.figure(1)
    # plt.plot(mean[:, 1], np.zeros(100), 'rx', alpha=0.5, label="Upper")
    # plt.plot(mean[:, 0], np.zeros(100), 'gx', alpha=0.5, label="Lower")
    # plt.xlabel('Confidence intervals values for mean')
    # plt.yticks([])
    # plt.legend()
    # plt.show()
    # plt.figure(2)
    # plt.plot(var[:, 1], np.zeros(100), 'ro', alpha=0.5, label="Upper")
    # plt.plot(var[:, 0], np.zeros(100), 'go', alpha=0.5, label="Lower")
    # plt.xlabel('Confidence intervals values for the variance')
    # plt.yticks([])
    # plt.legend()
    # plt.show()

    # histogram 3 for Pareto
    k = [2.05, 2.5, 3, 4]
    mean_list = []
    var_list = []
    dic = {"k:2.05": [], "k:2.5": [], "k:3": [], "k:4": []}
    for i in k:
        U = np.random.uniform(0.0, 0.1, 10000)
        par1, E, Var = funcPareto.pareto(U, i)
        ppf = np.linspace(pareto.ppf(0.01, i), pareto.ppf(0.99, i), 100)
        pdf = pareto.pdf(ppf, i)
        # comparison(par1, ppf-1, pdf, "Pareto comparison for k = {0}".format(i))
        dic["k:{0}".format(i)] = [E, Var, np.mean(par1), np.var(par1)]
    df = pd.DataFrame(dic, index=["Mean", "Variance", "Mean_analytical", "Variance_analytical"])
    print(df)

