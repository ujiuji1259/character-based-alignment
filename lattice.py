import math
from dictionary import PairDictionary
from language_model import Ngram

class Node(object):
    def __init__(self, min_cost, min_path, dst_str, min_surface):
        self.min_cost = min_cost
        self.min_path = min_path
        self.min_surface = min_surface
        self.dst_str = dst_str

class BOS(object):
    def __init__(self):
        self.pair_id = -1
        self.min_path = tuple()
        self.min_surface = (('', ''), )
        self.min_cost = 0
        self.dst_str = ''

class EOS(object):
    def __init__(self):
        self.pair_id = -2
        self.cost = 0

class Lattice(object):
    def __init__(self, size, dst, model, dst_list=None, normal_list=None):
        self.p = 1
        self.dst = dst
        self.model = model
        self.snodes = [[BOS()]] + [[] for i in range(0, size+1)]
        self.enodes = [[], [BOS()]] + [[] for i in range(0, size+1)]
        if dst_list is not None:
            self.dst_list = set(dst_list)
        else:
            self.dst_list = None

        if normal_list is not None:
            self.normal_list = set(normal_list)
        else:
            self.normal_list = None

    def add(self, idx, src, dst):
        min_cost = 214748364
        min_path = ()
        min_surface = ()
        min_dst = ''
        if self.p >= len(self.enodes) or not self.enodes[self.p]:
            return False

        for enode in self.enodes[self.p]:
            add_cost = 0
            dst_word = None
            dst_word = enode.dst_str + dst
            if self.dst and dst_word != self.dst[:len(dst_word)]:
                continue
            if self.dst_list and dst_word not in self.dst_list:
                continue

            if idx != -1:
                cost = enode.min_cost + self.model.get_probability(enode.min_path, idx) + add_cost
            else:
                cost = enode.min_cost + 100

            if cost < min_cost:
                min_cost = cost
                min_path = (*enode.min_path, idx)
                min_surface = (*enode.min_surface, (src, dst))
                min_dst = dst_word

        if min_path != ():
            node = Node(min_cost, min_path, min_dst, min_surface)
            self.snodes[self.p].append(node)
            self.enodes[self.p + len(src)].append(node)
        return True

    def forward(self):
        old_p = self.p
        self.p += 1
        while self.p < len(self.enodes) and not self.enodes[self.p]:
            self.p += 1

        if self.p == len(self.enodes):
            return False

        return self.p - old_p

    def end(self):
        min_cost = 214748364700
        min_path = ()
        min_surface = ()
        if self.p >= len(self.enodes) or not self.enodes[self.p]:
            return False, False, False

        for enode in self.enodes[self.p]:
            if self.dst and enode.dst_str != self.dst:
                continue
            if self.normal_list and enode.dst_str not in self.normal_list:
                continue

            cost = enode.min_cost

            if cost < min_cost:
                min_cost = cost
                min_path = enode.min_path
                min_surface = enode.min_surface

        return min_cost, min_path, min_surface


