import numpy as np
from torch.cuda import is_available
from string2string.similarity import BERTScore

class Bertscore:
    def __init__(self, model_name):
        self.bertscore = BERTScore(
            model_name_or_path=model_name,
            device='cuda' if is_available() else 'cpu'
            )
        self.metric_type = 'precision'

    def set_metric_type(self, metric_type):
        self.metric_type = metric_type

    def calc_matrix(self, group, url2record):
        n = len(group)

        headlines = [url2record[url]['patched_title'] for url in group]
        texts = [url2record[url]['patched_text'] for url in group]
        
        headlines = np.array(headlines).repeat(n).tolist()
        texts = texts * n

        matrix = self.bertscore.compute(headlines, texts)[self.metric_type].reshape(n, n)
                
        return matrix
