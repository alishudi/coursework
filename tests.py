import numpy as np

class Rand_metric:
    def __init__(self):
        pass

    def calc_matrix(self, group, url2record):
        n = len(group)
        return np.random.rand(n, n)
    

class Correct_metric:
    """
    An attempt to get a corret ranking from markups
    """
    def __init__(self):
        pass

    def calc_matrix(self, group, url2record, markup):
        n = len(group)
        matrix = np.zeros((n, n))
        
        url2ind = {}
        for j, url in enumerate(group):
            url2ind[url] = j

        for pair in markup:
            qual = pair['quality']
            if qual == 'left':
                matrix[url2ind[pair['right_url']], url2ind[pair['left_url']]] += 1
            elif qual == 'right':
                matrix[url2ind[pair['left_url']], url2ind[pair['right_url']]] += 1
                
        return matrix