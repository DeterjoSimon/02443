def TSP(A, S):
    cost = 0
    for i in range(len(S)-1):
        cost += A[S[i] - 1][S[i+1] - 1]
    return cost

