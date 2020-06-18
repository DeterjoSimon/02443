import numpy as np
import random
import scipy.stats
from Exercise5 import MonteCarlo, Antithetic, Control
import pandas as pd


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h


if __name__ == '__main__':
    pd.set_option('display.max_columns', 10)
    n = 100

    X, E, Var = MonteCarlo.Crude_Monte_Carlo(n)
    theo_E = 1.72
    theo_Var = 0.2420
    print("--- Monte Carlo estimation for exponential integral ---")
    dic = {"Estimate": [E, Var], "Theoretical": [theo_E, theo_Var], "CI": [mean_confidence_interval(X)]}
    df = pd.DataFrame(dic, index=["Mean", "Variance"])
    print(df)

    Y, E, Var = Antithetic.Antithetic_estimator(n)
    theo_Var = 0.0039
    print("--- Antithetic estimation for exponential integral ---")
    dic = {"Estimate": [E, Var], "Theoretical": [theo_E, theo_Var], "CI": [mean_confidence_interval(Y)]}
    df = pd.DataFrame(dic, index=["Mean", "Variance"])
    print(df)

    Z, E, Var = Control.control_estimate(n)
    print("--- Control estimation for exponential integral ---")
    dic = {"Estimate": [E, Var], "Theoretical": [theo_E, theo_Var], "CI": [mean_confidence_interval(Z)]}
    df = pd.DataFrame(dic, index=["Mean", "Variance"])
    print(df)

    W, E, Var = Control.control_estimate(n)
    print(W.size, " : Size of W")
    print("--- Stratified estimation for exponential integral ---")
    dic = {"Estimate": [E, Var], "Theoretical": [theo_E, theo_Var], "CI": [mean_confidence_interval(W)]}
    df = pd.DataFrame(dic, index=["Mean", "Variance"])
    print(df)
