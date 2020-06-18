import numpy as np
import math


def bootstrap_mean(initial_sample, n, epochs=100):
    res = [np.random.choice(initial_sample, n, replace=True) for _ in range(epochs)]
    return np.mean(res, axis=1), epochs


def bootstrap_var(initial_sample, n, epochs=100, var_estimate=[]):
    sample = [np.random.choice(initial_sample, n, replace=True) for _ in range(epochs)]
    for i in range(epochs):
        var_estimate.append(np.var(sample[i]))
    var_var_estimate = sum((var_estimate - np.var(initial_sample)) ** 2) / epochs
    return var_var_estimate, epochs


def bootstrap_med_mean(initial_sample, n, epochs=100, med_estimate=[], mean_estimate=[]):
    sample = [np.random.choice(initial_sample, n, replace=True) for _ in range(epochs)]
    for i in range(epochs):
        mean_estimate.append(np.mean(sample[i]))
        med_estimate.append(np.median(sample[i]))
    med_estimate = np.array(med_estimate)
    mean_estimate = np.array(mean_estimate)
    var_med_estimate = sum((med_estimate - np.median(initial_sample)) ** 2) / epochs
    var_mean_estimate = sum((mean_estimate - np.mean(initial_sample)) ** 2) / epochs
    return var_med_estimate, var_mean_estimate

