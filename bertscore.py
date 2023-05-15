import numpy as np
from torch.cuda import is_available
from string2string.similarity import BERTScore
from model_lists import model_lists

class Bertscore:
    def __init__(self, model_name, num_layers=None):
        self.bertscore = BERTScore(
            model_name_or_path=model_name,
            num_layers=num_layers,
            device='cuda' if is_available() else 'cpu'
            )
        self.metric_type = 'precision'
        
        #those models has wrong max sequence length assigned on HF
        if model_name in model_lists['512']:
            self.bertscore.tokenizer.model_max_length = 512

        if model_name in model_lists['1024']:
            self.bertscore.tokenizer.model_max_length = 1024

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

    def get_scores(self, group, url2record, best_headline):
        headlines = [url2record[url]['patched_title'] for url in group]
        reference_headline = url2record[best_headline]['patched_title']

        scores = self.bertscore.compute(headlines, [reference_headline] * len(headlines))

        return scores[self.metric_type]