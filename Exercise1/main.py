from Exercise1 import LCG as lcg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare


def histogram():
    n, bins, patches = plt.hist(x=U, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


if __name__ == '__main__':
    U = lcg.lcg(13, 911, 11584577, 3)
    print(chisquare(U))