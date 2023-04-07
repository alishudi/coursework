from navec import Navec
from string import punctuation
import numpy as np
import evaluate
import networkx as nx
from tqdm import tqdm, trange

navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
rouge = evaluate.load('rouge')

def tokenize(text): 
    words = [tok for tok in text.split(' ') if tok not in punctuation]
    return ' '.join([str(navec.vocab.get(word, navec.vocab.unk_id)) for word in words])


def calc_rouge_matrix(group, url2record, rouge_type='rouge1'):
    n = len(group)
    matrix = []

    headlines = [tokenize(url2record[url]['patched_title']) for url in group]
    texts = [tokenize(url2record[url]['patched_text']) for url in group]
    
    for i in range(n):
        matrix.append(rouge.compute(
            predictions=headlines[i:i+1] * n,
            references=texts,
            rouge_types=[rouge_type],
            use_aggregator=False
        )[rouge_type])
            
    return np.array(matrix)

def evaluate_rouge(groups, markups, url2record, rouge_type='rouge1', threshold=0.01):
    accs = []

    for i in trange(len(groups)):
        group = list(groups[i])
        matrix = calc_rouge_matrix(group, url2record, rouge_type=rouge_type)

        nx_graph = nx.from_numpy_array(matrix)
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
                accs.append(int(score_left >= score_right))
            elif qual == 'right':
                accs.append(int(score_right >= score_left))
            elif qual == 'draw':
                if np.abs(score_right - score_left) < threshold:
                    accs.append(1)
                else:
                    accs.append(0.5)
                    
    return np.mean(accs)