from navec import Navec
from string import punctuation
import numpy as np
import evaluate

class Rouge:
    def __init__(self, navec_model='navec_hudlit_v1_12B_500K_300d_100q.tar'):
        self.navec = Navec.load(navec_model)
        self.rouge = evaluate.load('rouge')

    def set_rouge_type(self, rouge_type):
        self.rouge_type = rouge_type

    def tokenize(self, text): 
        words = [tok for tok in text.split(' ') if tok not in punctuation]
        return ' '.join([str(self.navec.vocab.get(word, self.navec.vocab.unk_id)) for word in words])

    def calc_matrix(self, group, url2record):
        n = len(group)
        matrix = []

        headlines = [self.tokenize(url2record[url]['patched_title']) for url in group]
        texts = [self.tokenize(url2record[url]['patched_text']) for url in group]
        
        for i in range(n):
            matrix.append(self.rouge.compute(
                predictions=headlines[i:i+1] * n,
                references=texts,
                rouge_types=[self.rouge_type],
                use_aggregator=False
            )[self.rouge_type])
                
        return np.array(matrix)