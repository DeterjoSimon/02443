def I(x, x_i):
    return int(x_i <= x)


def Fe(data, x):
    F = 0
    for i in range(len(data)):
        F += I(x, data[i])
    F /= len(data)
    return F

