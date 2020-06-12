from scipy import stats
import numpy as np


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
    return Server.blocks


def arrival_Poisson(n_customers, mean_time_btw):
    result = stats.expon.rvs(size=n_customers, scale=mean_time_btw)
    return result


def arrival_Erlang(n_customers, mean_time_btw):
    result = stats.erlang.rvs(1, size=n_customers, scale=mean_time_btw)
    return result


def service_Exponential(n_customers, mean_serv_time):
    result = stats.expon.rvs(size=n_customers, scale=mean_serv_time)
    return result


def event_poisson_exponential(S, n_c, n_sim, mean_time_btw, mean_serv_time):
    return process(S, n_c, arrival_Poisson(n_c, mean_time_btw), service_Exponential(n_c, mean_serv_time))


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
        result.append(event_poisson_exponential(S, n_costumers, n_simulations, mean_time_btw, mean_serv_time))
    print(result)
    print(sum(result)/n_costumers)

