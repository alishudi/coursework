from navec import Navec
from string import punctuation
import numpy as np
import evaluate
import multiprocessing as mp
from functools import partial

class Bleu:
    def __init__(self, navec_model='navec_hudlit_v1_12B_500K_300d_100q.tar'):
        self.navec = Navec.load(navec_model)
        self.bleu = evaluate.load('bleu')
        self.max_n = 1
        
    def set_max_n(self, max_n):
        self.max_n = max_n

    def tokenize(self, text): 
        words = [tok for tok in text.split(' ') if tok not in punctuation]
        return ' '.join([str(self.navec.vocab.get(word, self.navec.vocab.unk_id)) for word in words])

    def compute_bleu(self, headlines, texts, n, i):
            return self.bleu.compute(
                predictions=[headlines[i // n]],
                references=[texts[i % n]],
                max_order=self.max_n
            )['bleu']
        
    def calc_matrix(self, group, url2record):
        n = len(group)

        headlines = [self.tokenize(url2record[url]['patched_title']) for url in group]
        texts = [self.tokenize(url2record[url]['patched_text']) for url in group]
        
            
        #i couldn't find any parallelized bleu implementation so had to use loop
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(partial(self.compute_bleu, headlines, texts, n), range(n**2))
        matrix = np.array(results).reshape(n, n)
        
        return matrix

    def get_scores(self, group, url2record, best_headline):
        headlines = [self.tokenize(url2record[url]['patched_title']) for url in group]
        reference_headline = self.tokenize(url2record[best_headline]['patched_title'])

        scores = []
        for i in range(len(headlines)):
            scores.append(self.bleu.compute(
                predictions=headlines[i:i+1],
                references=[reference_headline],
                max_order=self.max_n
            )['bleu'])
        return scores