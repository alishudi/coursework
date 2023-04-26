import numpy as np
from evaluate import load
import networkx as nx
from tqdm import tqdm, trange

bertscore = load("bertscore")


def calc_bertscore_matrix(group, url2record, model_type, metric_type='precision'):
    n = len(group)
    matrix = []

    headlines = [url2record[url]['patched_title'] for url in group]
    texts = [url2record[url]['patched_text'] for url in group]
    
    for i in range(n):
        matrix.append(bertscore.compute(
            predictions=headlines[i:i+1] * n,
            references=texts,
            model_type=model_type
        )[metric_type])
            
    return np.array(matrix)

def evaluate_bertscore(groups, markups, url2record, model_type, metric_type='precision', threshold=0.01):
    accs = []

    for i in trange(len(groups)):
        group = list(groups[i])
        matrix = calc_bertscore_matrix(group, url2record, model_type,  metric_type=metric_type)

        nx_graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        scores = nx.pagerank(nx_graph)

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
            elif qual == 'draw':
                if np.abs(score_right - score_left) < threshold:
                    accs.append(1)
                else:
                    accs.append(0.5)
                    
    return np.mean(accs)