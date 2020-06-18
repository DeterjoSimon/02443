import numpy as np


def control_estimate(n):
    U = np.random.uniform(0, 1, n)
    X = np.exp(U)
    Y = U
    meanX = sum(X)/n
    meanY = sum(Y)/n
    varY = sum(Y**2)/n - meanY**2
    varX = sum(X**2)/n - meanX**2
    # Cov(X,Y)= E[(X−EX)(Y−EY)]=E[XY]−(EX)(EY)
    covXY = np.dot(X, Y)/n - (sum(X)/n) * meanY
    c = - covXY / varY
    Z = X + c*(Y - meanY)
    meanZ = sum(Z)/n
    varZ = varX - covXY**2/varY
    return Z, meanZ, varZ

