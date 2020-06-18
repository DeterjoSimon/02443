from Exercise8 import Bootstrap
import numpy as np
from Exercise3 import funcPareto


def subroutine(beta=1, k=1.05, N=200, r=100):
    res, E, Var = funcPareto.pareto(k, N, beta)
    var_med_estimate, var_mean_estimate = Bootstrap.bootstrap_med_mean(res, N, epochs=r)
    return var_med_estimate, var_mean_estimate, np.mean(res, axis=0), np.median(res)


if __name__ == "__main__":
    a = -5
    b = 5
    print("--- Bootstrap to estimate p ---")
    Xi = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
    res, reps = Bootstrap.bootstrap_mean(Xi, 10)
    mu = np.mean(Xi)
    p = sum([a < estimate < b for estimate in (res - mu)]) / reps
    print("Estimate of p: {0}\n".format(p*100))

    print("--- Bootstrap to estimate Var(S^2) ---")
    Xi = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8])
    n = 15
    res, reps = Bootstrap.bootstrap_var(Xi, n)
    print("The bootstrap estimate of Var(S^2) is: {0}\n".format(res))

    print("--- Bootstrap estimate of mean and variance for Pareto distribution ---")
    var_med_estimate, var_mean_estimate, mean_sample, med_sample = subroutine()
    print("The sample mean and median are respectively: {0} and {1}\n".format(mean_sample, med_sample))
    print("The bootstrap estimate of the variance of the sample mean is: {0}\n".format(var_mean_estimate))
    print("The bootstrap estimate of the variance of the sample median is: {0}\n".format(var_med_estimate))

    print("---------------------------------------------------------------------\n")
