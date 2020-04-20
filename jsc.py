from dictionary import PairDictionary
from language_model import Ngram
from lattice import Lattice
import os
from tqdm import tqdm

class JSC(object):
    def __init__(self, path, n):
        self.data_dir = path
        self.dic = PairDictionary(path=self.data_dir)
        self.model = Ngram(n, self.dic)
        self.model.build_vocab(os.path.join(self.data_dir, 'pair_vocab.txt'))
        self.dst_list = None
        self.normal_list = None

    def create_dst_list(self, path):
        with open(path, 'r') as f:
            lines = [line.split('\t') for line in f.read().split('\n') if line != '']

        dst = [line[1] for line in lines]
        self.normal_list = dst
        print(self.normal_list)
        self.dst_list = ['']
        for d in dst:
            for i in range(len(d)):
                for j in range(i+1, len(d)+1):
                    self.dst_list.append(d[i:j])


    def load_trained_file(self):
        self.model.load_parameter(self.data_dir)

    def decode(self, sent, dst=None):
        lattice = Lattice(len(sent), dst, self.model, self.dst_list, self.normal_list)
        i = 0
        while i < len(sent):
            word = sent[i:]
            res = self.dic.common_prefix_search(word)
            res = [(r, self.dic.pair_itop[r].src, self.dic.pair_itop[r].dst) for r in res]
            res.append((self.model.UNK_id, word[0], word[0]))

            for r in res:
                lattice.add(*r)

            p = lattice.forward()
            i += p


        cost, path, surface = lattice.end()
        if not path:
            return sent, '', [], None

        src = ' '.join([s[0] for s in surface])
        dst = ' '.join([s[1] for s in surface])

        return src[1:], dst[1:], path, cost

    def train(self, path):
        with open(path, 'r') as f:
            lines = [line.split('\t') for line in f.read().split('\n') if line != '']

        cost_his = [10000000000, 1000000000]
        
        while cost_his[-1] < cost_his[-2]:
            ngrams = []
            all_cost = 0
            for line in tqdm(lines):
                src = line[0]
                dst = line[1]
                src_, dst_, path, cost = self.decode(src, dst)
                if cost is not None:
                    all_cost += cost
                ngrams.append(path)
            print(all_cost)

            self.model.train(ngrams)
            cost_his.append(all_cost)
        self.model.save(self.data_dir)

if __name__ == '__main__':
    jsc = JSC('vocab', 1)
    #jsc.load_trained_file()
    print('finish_load')
    jsc.create_dst_list('sample.txt')
    print(jsc.decode('大腸性下痢便'))
    print(jsc.decode('細菌性赤痢'))
    print(jsc.decode('肺外結核'))
    print(jsc.decode('結核病変', '結核'))
    jsc.train('sample.txt')
    print('finish training')
    while True:
        word = input()
        print(jsc.decode(word))
