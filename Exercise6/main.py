from math import factorial
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def P(i, j, A1, A2):
    K = 1
    return (A1 ** i / factorial(i)) * (A2 ** j / factorial(j))


def metropolis_hastings(A, n):
    Xi = np.zeros(n)
    for i in range(1, n - 1):
        current = Xi[i - 1]
        Yi = np.random.randint(0, 11)

        gy = (A ** Yi) / factorial(Yi)
        gx = (A ** current) / factorial(current)
        if np.random.uniform(0, 1) <= min(1, gy / gx):
           Xi[i + 1] = Yi
        else:
            Xi[i + 1] = current
    return Xi


def metropolis_hastings_direct(A1, A2, n):
    Xi = np.zeros(n)
    Yi = np.zeros(n)

    for i in range(1, n - 1):
        x, new_x = Xi[i - 1], np.random.randint(0, 11)
        y, new_y = Yi[i - 1], np.random.randint(0, 11)

        gy = P(new_x, new_y, A1, A2)
        gx = P(x, y, A1, A2)

        if np.random.uniform(0, 1) <= min(1, gy / gx):
            Xi[i+1] = new_x
            Yi[i+1] = new_y
        else:
            Xi[i+1] = x
            Yi[i+1] = y
    return Xi, Yi


def actual_densities():
    n = 10
    A = 8
    d = []
    t = []
    for j in range(n):
        d.append(A ** j / factorial(j))
    for i in range(n):
        t.append((A ** i / factorial(i)) / sum(d))

    return t


if __name__ == "__main__":
    A = 1 * 8
    n = 10
    temp = np.array([A ** i / factorial(i) for i in range(n)])
    Y = [(A ** i)/factorial(i)/sum(temp) for i in range(n)]
    X = metropolis_hastings(A, 10000)
    plt.hist(X, bins=10)
    plt.show()
    print(X[1:1000])
    print("--- Chi-squared test ---")
    print(scipy.stats.chisquare(np.histogram(X, 10, density=True)[0], f_exp=Y))

    print("\n--- Coordinated Metropolis  ---")
    X, YY = metropolis_hastings_direct(4, 4, 10000)
    print(X, " :X \n")
    print(YY, " :Y \n")

    print("--- Chi-squared test ---")
    print("--- For X ---")
    print(scipy.stats.chisquare(np.histogram(X, 10, density=True)[0], f_exp=Y))
    print("\n--- For Y ---")
    print(scipy.stats.chisquare(np.histogram(YY, 10, density=True)[0], f_exp=Y))

