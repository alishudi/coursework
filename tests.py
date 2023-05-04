import numpy as np

class Rand_metric:
    def __init__(self):
        pass

    def calc_matrix(self, group, url2record):
        n = len(group)
        return np.random.rand(n, n)