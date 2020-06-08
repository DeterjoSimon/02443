from math import floor
from math import log
import numpy as np


def geometric(U, p):
    temp = np.log(U)/log(1-p)
    return np.floor(temp) + 1
