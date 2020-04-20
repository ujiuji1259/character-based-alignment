import dartsclone
import os

class PairDictionary(object):
    def __init__(self, **kwarg):
        if 'path' in kwarg:
            self._load_from_dir(kwarg['path'])
        else:
            self.src_itos = kwarg['src_vocab']
            self.src_stoi = {s:i for i, s in enumerate(self.src_itos)}
            self.src_trie = kwarg['src']
            self.offset = kwarg['offset']
            self.pair_itop = kwarg['pair']
            self.pair_stoi = {str(p):i for i, p in enumerate(self.pair_itop)}

    def __len__(self):
        return len(self.pair_itop)

    def _load_from_dir(self, directory):
        with open(os.path.join(directory, 'src_vocab.txt'), 'r') as f:
            self.src_itos = [line.split('\t')[1] for line in f.read().split('\n') if line != '']
        self.src_stoi = {s:i for i, s in enumerate(self.src_itos)}

        self.src_trie = dartsclone.DoubleArray()
        self.src_trie.open(os.path.join(directory, 'src.dic'))

        with open(os.path.join(directory, 'offset.txt'), 'r') as f:
            self.offset = [line.split() for line in f.read().split('\n') if line != '']
            self.offset = {int(i):(int(i2), int(i3)) for i, i2, i3 in self.offset}

        with open(os.path.join(directory, 'pair_vocab.txt'), 'r') as f:
            self._create_pair_vocab([line.split('\t')[0] for line in f.read().split('\n') if line != ''])
            self.pair_stoi = {str(p):i for i, p in enumerate(self.pair_itop)}


    def _create_pair_vocab(self, pair):
        self.pair_itop = [p.split('/') for p in pair]
        self.pair_itop = [PairEntry(p[0], p[1], self.src_stoi[p[0]]) for p in self.pair_itop]

    def decode(self, ids):
        return ' '.join([self.pair_itop[i].src for i in ids]), ' '.join([self.pair_itop[i].dst for i in ids])

    def common_prefix_search(self, word, src=False, pair=True):
        res = self.src_trie.common_prefix_search(word.encode('utf-8'), pair_type=False)

        tmp = [self.offset[r] for r in res]
        tmp = [list(range(t[0], t[1])) for t in tmp]
        return sum(tmp, [])

class PairEntry(object):
    def __init__(self, src, dst, src_idx):
        self.src = src
        self.dst = dst
        self.src_idx = src_idx

    def __eq__(self, other):
        if other is None or not isinstance(other, PairEntry):
            return False
        return self.src == other.src and self.dst == other.dst

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.src + '/' + self.dst

    def __len__(self):
        return 1

    def __add__(self, other):
        return PairEntry('', '', self.cost + other.cost)

    def hash(self):
        return hash(self.src + '/' + self.dst)

if __name__ == '__main__':
    dic = PairDictionary(path='vocab')
    print(dic.common_prefix_search('ショ'))


