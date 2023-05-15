from navec import Navec
from string import punctuation
import numpy as np
import evaluate

class Rouge:
    def __init__(self, navec_model='navec_hudlit_v1_12B_500K_300d_100q.tar'):
        self.navec = Navec.load(navec_model)
        self.rouge = evaluate.load('rouge')
        self.rouge_type = 'rouge1'

    def set_rouge_type(self, rouge_type):
        self.rouge_type = rouge_type

    def tokenize(self, text): 
        words = [tok for tok in text.split(' ') if tok not in punctuation]
        return ' '.join([str(self.navec.vocab.get(word, self.navec.vocab.unk_id)) for word in words])

    def calc_matrix(self, group, url2record):
        n = len(group)

        headlines = [self.tokenize(url2record[url]['patched_title']) for url in group]
        texts = [self.tokenize(url2record[url]['patched_text']) for url in group]
        
        headlines = np.array(headlines).repeat(n).tolist()
        texts = texts * n
        

        matrix = self.rouge.compute(
            predictions=headlines,
            references=texts,
            rouge_types=[self.rouge_type],
            use_aggregator=False
        )[self.rouge_type]
                
        return np.array(matrix).reshape(n, n)
    
    def get_scores(self, group, url2record, best_headline):
        headlines = [self.tokenize(url2record[url]['patched_title']) for url in group]
        reference_headline = self.tokenize(url2record[best_headline]['patched_title'])

        scores = self.rouge.compute(
            predictions=headlines,
            references=[reference_headline] * len(headlines),
            rouge_types=[self.rouge_type],
            use_aggregator=False
        )[self.rouge_type]

        return scores