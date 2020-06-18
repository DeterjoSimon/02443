import numpy as np
from Exercise7 import TSP
import mlrose
import os
import pandas as pd


def distance_maker(Mcost):
    dist_list = []
    for i in range(len(Mcost)):
        for j in range(len(Mcost)):
            if i == j:
                continue
            else:
                dist_list.append((i, j, Mcost.iloc[i][j]))
    return dist_list


if __name__ == "__main__":
    data = pd.read_csv(filepath_or_buffer="C:/Users/Simon/Desktop/AI/Assignment/cost.csv", header=0, index_col=0)
    # Test Travelling salesman
    A = np.array([[0, 5, 3, 1, 4, 12],
                  [2, 0, 22, 11, 13, 30],
                  [6, 8, 0, 13, 12, 5],
                  [33, 9, 5, 0, 60, 17],
                  [1, 15, 6, 10, 0, 14],
                  [24, 6, 8, 9, 40, 0]])
    S = [1, 2, 3, 4, 5, 6, 1]
    # print(TSP.TSP(A, S))
    print(distance_maker(data))

