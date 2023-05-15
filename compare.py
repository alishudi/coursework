import numpy as np
import networkx as nx
from tqdm import trange

class pagerank_scorer:
    def __init__(self, similiarity_metric):
        self.similiarity_metric = similiarity_metric

    def get_scores(self, group, url2record, best_headline):
            matrix = self.similiarity_metric.calc_matrix(group, url2record)
            nx_graph = nx.from_numpy_array(matrix.T, create_using=nx.DiGraph)
            scores = nx.pagerank(nx_graph, max_iter=200)
            return scores

def compare_metric(groups, markups, url2record, scorer, best_headlines, nont_clusters):

    accs = []
    for i in trange(len(groups)):
        if i in nont_clusters:
            continue
        
        #if there are several headlines tied for 1st place then will calculate for both and take average
        local_accs = []
        calc_counts = []
        for pair in markups[i]:
            if pair['left_url'] in best_headlines[i] or pair['right_url'] in best_headlines[i]:
                m = len(best_headlines[i]) #number of times we using this pair to calculate accuracy
            else:
                m = len(best_headlines[i]) - 1
            calc_counts.append(m)
            local_accs.append([])
        
        for best_headline in best_headlines[i]:
            group = [url for url in groups[i] if url != best_headline]
            
            scores = scorer.get_scores(group, url2record, best_headline)

            url2ind = {}
            for j, url in enumerate(group):
                url2ind[url] = j

            for ind, pair in enumerate(markups[i]):
                if pair['left_url'] == best_headline or pair['right_url'] == best_headline:
                    continue   
                qual = pair['quality']
                score_left = scores[url2ind[pair['left_url']]]
                score_right = scores[url2ind[pair['right_url']]]
                if qual == 'left':
                    local_accs[ind].append(int(score_left > score_right))
                elif qual == 'right':
                    local_accs[ind].append(int(score_right > score_left))
        local_accs = [np.mean(local_acc) for local_acc in local_accs if len(local_acc) > 0]
        accs = np.append(accs, np.array(local_accs))
    
    return np.mean(accs)