from collections import Counter
import math
import os
import itertools
from dictionary import PairDictionary

class NgramEntry(object):
    def __init__(self, pairs, cost):
        self.cost = cost
        self.pair_ids = tuple(pairs)

    def get_marginal(self):
        if len(self.pair_ids) == 1:
            return (0,)
        else:
            return tuple(self.pair_ids[:-1])

    def __str__(self):
        return ' '.join([str(p) for p in self.pair_ids])

class Ngram(object):
    def __init__(self, n, pair_dic):
        self.n = n
        self.dic = pair_dic
        self.BOS_id = len(self.dic)
        self.UNK_id = self.BOS_id + 1

    def save(self, path):
        output = []
        for n in self.ngram_iton.values():
            ngram = [str(r) for r in n.pair_ids]
            cost = n.cost
            output.append("\t".join([*ngram, str(cost)]))
        with open(os.path.join(path, 'trained_ngram_cost.txt'), 'w') as f:
            f.write('\n'.join(output))

    def load_parameter(self, path):
        with open(os.path.join(path, 'trained_ngram_cost.txt'), 'r') as f:
            lines = [line.split('\t') for line in f.read().split('\n') if line != '']
        for line in lines:
            ids, cost = line[:-1], float(line[-1])
            ids = tuple([int(i) for i in ids])
            self.ngram_iton[ids].cost = cost

        self._calc_marginal_cost()
        self.prob = {}
        self.set_prob()

    def build_vocab(self, fn):
        def calc_total_costs(id_list, cost_dic):
            id_list = [cost_dic[i] for i in id_list]
            return sum(id_list)

        with open(fn, 'r') as f:
            lines = [line.split('\t') for line in f.read().split('\n') if line != '']
        unigram = [(self.dic.pair_stoi[line[0]],)  for line in lines] + [(self.UNK_id,)]
        costs = {self.dic.pair_stoi[line[0]]:int(line[1]) for line in lines}
        costs[self.UNK_id] = 0
        costs[self.BOS_id] = 0
        vocab = unigram[:]

        if self.n != 1:
            for i in range(self.n-1):
                vocab = vocab + [(*i[0], *i[1]) for i in list(itertools.product(vocab, unigram))]

        vocab = [tuple([self.BOS_id]*(self.n-len(i)) + list(i)) for i in vocab]
        self.ngram_iton = {i:NgramEntry(i, calc_total_costs(i, costs)) for i in vocab}
        self.marginal_iton = {n.get_marginal():NgramEntry(n.get_marginal(), 0) for n in self.ngram_iton.values()}

        self._calc_marginal_cost()
        self.prob = {}
        for i, n in self.ngram_iton.items():
            self.prob[i] = calc_total_costs(i, costs)

    def _calc_marginal_cost(self):
        for n in self.ngram_iton.values():
            self.marginal_iton[n.get_marginal()].cost += n.cost

    @staticmethod
    def smoothing(u, d, la, unify):
        if d == 0:
            divided = 0
        else:
            divided = u / d
        return la * divided + (1 - la) * unify

    def set_prob(self):
        for i, n in self.ngram_iton.items():
            m = self.marginal_iton[n.get_marginal()].cost
            self.prob[i] = -math.log10(self.smoothing(n.cost, m, 0.9, 1e-5))

    def get_probability(self, ids, next_id):
        tokens = tuple([self.BOS_id for i in range(self.n-1)] + list(ids) + [next_id])
        return self.prob[tokens[-self.n:]]

    def train(self, token_list):
        self.clear()
        for tokens in token_list:
            tokens = tuple([self.BOS_id]*(self.n - 1) + list(tokens))
            for i in range(self.n, len(tokens)+1):
                n = self.ngram_iton[tokens[i-self.n:i]]
                n.cost += 1
                self.marginal_iton[n.get_marginal()].cost += 1
        self.set_prob() 

    def EM_train(self, token_list):
        self.clear()
        for tokens in token_list:
            tokens = tuple([self.BOS_id]*(self.n-1) + list(tokens))
            for i in range(self.n, len(tokens)+1):
                n = self.ngram_iton[tokens[i-self.n:i]]
                n.cost += self.prob[n.pair_ids]
                self.marginal_iton[n.get_marginal()].cost += self.prob[n.pair_ids]
        self.set_prob() 

    def clear(self):
        for n in self.ngram_iton.values():
            n.cost = 0
        for n in self.marginal_iton.values():
            n.cost = 0

        for n in self.ngram_iton.values():
            n.cost = 0

if __name__ == '__main__':
    dic = PairDictionary(path='vocab')
    model = Ngram(2, dic)

    model.build_vocab('vocab/pair_vocab.txt')
    print(model.get_probability([1,2,3], 4))
    eg = [[1,2,3],
            [2,3,4],
            [3,5,5],
            [3,4,4]]
    model.train(eg)
    print(model.get_probability([1,2,3], 4))
    #ngram = Ngram(3, vocab)

