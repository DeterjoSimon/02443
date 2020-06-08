import numpy as np


def lcg(a, c, M, x_0, length=10000):
    x = np.zeros(length)
    x[0] = x_0
    for i in range(1, length):
        x[i] = (a * x[i - 1] + c) % M

    return x


if __name__ == '__main__':
    U = lcg(13, 911, 11584577, 3)
