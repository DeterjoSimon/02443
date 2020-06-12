
def pareto(U, k, beta=1):
    X = beta * (U**(-1/k))
    E = (k/(k-1)) * beta
    Var = (k/((k-1)**2)*(k-2)) * beta**2
    return X, E, Var
