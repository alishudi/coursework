from navec import Navec
from string import punctuation
import numpy as np
import evaluate
import multiprocessing as mp
from functools import partial

class Chrf:
    def __init__(self, tokenize=False, navec_model='navec_hudlit_v1_12B_500K_300d_100q.tar'):
        if tokenize:
             self.navec = Navec.load(navec_model)
        self.tok = tokenize
        self.chrf = evaluate.load('chrf')
        self.word_order = 0
        
    def set_word_order(self, word_order):
        self.word_order = word_order

    def tokenize(self, text): 
        words = [tok for tok in text.split(' ') if tok not in punctuation]
        return ' '.join(words)

    def compute_chrf(self, headlines, texts, n, i):
            return self.chrf.compute(
                predictions=[headlines[i // n]],
                references=[texts[i % n]],
                word_order=self.word_order
            )['score']
        
    def calc_matrix(self, group, url2record):
        n = len(group)

        if self.tok:
            headlines = [self.tokenize(url2record[url]['patched_title']) for url in group]
            texts = [self.tokenize(url2record[url]['patched_text']) for url in group]
        else:
            headlines = [url2record[url]['patched_title'] for url in group]
            texts = [url2record[url]['patched_text'] for url in group]
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(partial(self.compute_chrf, headlines, texts, n), range(n**2))
        matrix = np.array(results).reshape(n, n)
        
        return matrix

    def get_scores(self, group, url2record, best_headline):
        if self.tok:
            headlines = [self.tokenize(url2record[url]['patched_title']) for url in group]
            reference_headline = self.tokenize(url2record[best_headline]['patched_title'])
        else:
            headlines = [url2record[url]['patched_title'] for url in group]
            reference_headline = url2record[best_headline]['patched_title']

        scores = []
        for i in range(len(headlines)):
            scores.append(self.chrf.compute(
                predictions=headlines[i:i+1],
                references=[reference_headline],
                word_order=self.word_order
            )['score'])
        return scores