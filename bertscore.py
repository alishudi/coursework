import numpy as np
from evaluate import load

class Bertscore:
    def __init__(self, navec_model='navec_hudlit_v1_12B_500K_300d_100q.tar'):
        self.bertscore = load("bertscore")
        self.metric_type='precision'

    def set_model_type(self, model_type):
        self.model_type = model_type

    def set_metric_type(self, metric_type):
        self.metric_type = metric_type

    def calc_matrix(self, group, url2record):
        n = len(group)
        matrix = []

        headlines = [url2record[url]['patched_title'] for url in group]
        texts = [url2record[url]['patched_text'] for url in group]
        
        for i in range(n):
            matrix.append(self.bertscore.compute(
                predictions=headlines[i:i+1] * n,
                references=texts,
                model_type=self.model_type
            )[self.metric_type])
                
        return np.array(matrix)