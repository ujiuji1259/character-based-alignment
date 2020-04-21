from dictionary import PairDictionary
from language_model import Ngram
from lattice import Lattice
import os
from tqdm import tqdm
import argparse

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

    def predict(self, fn):
        with open(fn, 'r') as f:
            lines = [line.split('\t') for line in f.read().split('\n') if line != '']
            true = [line[0] for line in lines]
            src = [line[1] for line in lines]

        output = ['出現形\t正解\t予測']
        total = 0
        cnt = 0
        for t, s in tqdm(zip(true, src)):
            src_, dst_, path, cost = self.decode(s)
            dst_ = dst_.replace(' ', '')
            output.append('\t'.join([s, t, dst_]))

            if dst_ == t:
                cnt += 1
            total += 1
        print(cnt / total)
        with open('result.txt', 'w') as f:
            f.write('\n'.join(output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build model files')
    parser.add_argument('--vocab', type=str, help="specify vocab file")
    parser.add_argument('--ngram', type=int, help="specify ngram length")
    parser.add_argument('--train', type=str, help="specify train file")
    parser.add_argument('--load', action="store_true", help="specify train file")
    args = parser.parse_args()
    jsc = JSC(args.vocab, args.ngram)
    if args.load:
        jsc.load_trained_file()
    print('finish_load')
    jsc.create_dst_list(args.train)
    """
    jsc.predict('test.txt')

    jsc.train(args.train)
    """
    print('finish training')
    while True:
        word = input()
        print(jsc.decode(word))
