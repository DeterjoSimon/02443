from typing import Any, Union

from scipy import stats
import numpy as np
from Exercise3 import funcPareto
import scipy.stats
from Exercise5 import Control

class Server:

    def __init__(self, n_simulations):
        self.room = np.full((n_simulations, 2), 0, dtype=float)
        self.blocks = 0

    def available(self, index):
        if self.room[index][0] != 0.0:
            return False
        else:
            return True

    def enter(self, time_arrival):
        flag = 0
        for i in range(len(self.room)):
            if Server.available(self, i):
                flag = 1
                self.room[i][0] = 1
                self.room[i][1] = time_arrival
                break
        if flag == 0:
            Server.blocked(self)

    def blocked(self):
        self.blocks += 1

    def check_time(self, time):
        for i in range(len(self.room)):
            if time > self.room[i][1]:
                self.room[i] = [0, 0]

    def remove(self, index):
        self.room[index] = [0, 0]


def process(Server, n_costumers, arrival_time, service_time):
    t = np.cumsum(arrival_time, dtype=float)
    for i in range(n_costumers):
        Server.check_time(t[i])
        Server.enter(t[i] + service_time[i])
    return Server.blocks/n_costumers

    # ----------------------------------------------- #


def arrival_Poisson(n_customers, mean_time_btw):
    result = stats.expon.rvs(size=n_customers, scale=mean_time_btw)
    return result


def arrival_Erlang(n_customers, mean_time_btw):
    result = stats.erlang.rvs(1, size=n_customers, scale=mean_time_btw)
    return result


def arrival_HyperExp(n_customers):
    U = np.random.uniform(size=n_costumers)
    li = []
    for k in U:
        if k > 0.8:
            li.append(stats.expon.rvs(scale=1 / 0.8333))
        else:
            li.append(stats.expon.rvs(scale=1 / 5))
    return li

    # ---------------------------------------------- #


def service_Exponential(n_customers, mean_serv_time):
    result = stats.expon.rvs(size=n_customers, scale=mean_serv_time)
    return result


def service_Constant(n_customers, mean_serv_time):
    return [mean_serv_time for _ in range(n_customers)]


def service_Pareto(n_customers, k, mean_serv_time):
    U = np.random.uniform(0, 1, n_costumers)
    X, E, V = funcPareto.pareto(k, n_costumers)
    X = (k / (k - 1)) * mean_serv_time * X
    return X
    # ------------------------------------------------- #


def event_poisson_exponential(S, n_c, mean_time_btw, mean_serv_time):
    return process(S, n_c, arrival_Poisson(n_c, mean_time_btw), service_Exponential(n_c, mean_serv_time))


def event_erlang_exponential(S, n_c, mean_time_btw, mean_serv_time):
    return process(S, n_c, arrival_Erlang(n_c, mean_time_btw), service_Exponential(n_c, mean_serv_time))


def event_hyperexp_exponential(S, n_c, mean_serv_time):
    return process(S, n_c, arrival_HyperExp(n_c), service_Exponential(n_c, mean_serv_time))


def event_poisson_constant(S, n_c, mean_time_btw, mean_serv_time):
    return process(S, n_c, arrival_Poisson(n_c, mean_time_btw), service_Constant(n_c, mean_serv_time))


def event_poisson_pareto(S, n_c, mean_time_btw, mean_serv_time, k):
    return process(S, n_c, arrival_Poisson(n_c, mean_time_btw), service_Pareto(n_c, k, mean_serv_time))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


if __name__ == "__main__":
    mean_serv_time = 8
    mean_time_btw = 1
    erlangs = 8
    n_costumers = 10000
    n_simulations = 10
    m = 10
    S = Server(n_simulations)

    # Poisson - exponential
    np.random.seed(1)
    result = []
    for i in range(m):
        S = Server(n_simulations)
        result.append(event_poisson_exponential(S, n_costumers, mean_time_btw, mean_serv_time))
    print(result)
    mu, low, up = mean_confidence_interval(result)
    print("--- CI ---")
    print("Mean value: ", mu)
    print("[", low, " ; ", up, "]\n")

    result = []
    for i in range(m):
        S = Server(n_simulations)
        result.append(event_erlang_exponential(S, n_costumers, mean_time_btw, mean_serv_time))
    print(result)
    mu, low, up = mean_confidence_interval(result)
    print("--- CI ---")
    print("Mean value: ", mu)
    print("[", low, " ; ", up, "]\n")

    result = []
    for i in range(m):
        S = Server(n_simulations)
        result.append(event_hyperexp_exponential(S, n_costumers, mean_time_btw))
    print(result)
    mu, low, up = mean_confidence_interval(result)
    print("--- CI ---")
    print("Mean value: ", mu)
    print("[", low, " ; ", up, "]\n")

    result = []
    for i in range(m):
        S = Server(n_simulations)
        result.append(event_poisson_constant(S, n_costumers, mean_time_btw, mean_serv_time))
    print(result)
    mu, low, up = mean_confidence_interval(result)
    print("--- CI ---")
    print("Mean value: ", mu)
    print("[", low, " ; ", up, "]\n")

    result = []
    for i in range(m):
        S = Server(n_simulations)
        result.append(event_poisson_pareto(S, n_costumers, mean_time_btw, mean_serv_time, k=1.05))
    print(result)
    mu, low, up = mean_confidence_interval(result)
    print("--- CI ---")
    print("Mean value: ", mu)
    print("[", low, " ; ", up, "]\n")

    result = []
    for i in range(m):
        res, _, _ = Control.control_estimate(n_costumers)
        S = Server(n_simulations)
        result.append(event_poisson_exponential(S, n_costumers, np.mean(res), mean_serv_time))
    print(result)
    mu, low, up = mean_confidence_interval(result)
    print("--- CI ---")
    print("Mean value: ", mu)
    print("[", low, " ; ", up, "]\n")