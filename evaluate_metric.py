import numpy as np
import networkx as nx
from tqdm import trange

from lexrank_centrality import degree_centrality_scores


def evaluate_metric(groups, markups, url2record, similiarity_metric, rank_method, matrix_func=lambda x: x, dir=True, threshold=0.01):
    accs = []

    for i in trange(len(groups)):
        group = list(groups[i])
        matrix = similiarity_metric.calc_matrix(group, url2record)
        matrix = matrix_func(matrix)

        if rank_method == 'pagerank':
            if dir:
                nx_graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            else:
                nx_graph = nx.from_numpy_array(matrix)
            scores = nx.pagerank(nx_graph)
        elif rank_method == 'lexrank':
            scores = degree_centrality_scores(matrix, threshold=None)
        else:
            raise(NotImplementedError)

        url2ind = {}
        for j, url in enumerate(group):
            url2ind[url] = j

        for markup in markups[i]:
            qual = markup['quality']
            assert qual in ['left', 'right', 'draw']
            score_left = scores[url2ind[markup['left_url']]]
            score_right = scores[url2ind[markup['right_url']]]
            if qual == 'left':
                accs.append(int(score_left > score_right))
            elif qual == 'right':
                accs.append(int(score_right > score_left))
            # elif qual == 'draw':
            #     if np.abs(score_right - score_left) < threshold:
            #         accs.append(1)
            #     else:
            #         accs.append(0.5)
                    
    return np.mean(accs)