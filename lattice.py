import math
from dictionary import PairDictionary
from language_model import Ngram
import heapq

class Node(object):
    def __init__(self, min_cost, min_path, dst_str, min_surface):
        self.min_cost = min_cost
        self.min_path = min_path
        self.min_surface = min_surface
        self.dst_str = dst_str

    def __comp__(self, other):
        return self.min_cost < ohter.min_cost

    def __eq__(self, other):
        return self.min_cost == other.min_cost

class BOS(Node):
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
    def __init__(self, size, dst, model, dst_list=None, normal_list=None, nbest=None):
        self.p = 1
        self.dst = dst
        self.model = model

        self.nbest = nbest
        if nbest is None:
            self.snodes = [[BOS()]] + [[] for i in range(0, size+1)]
            self.enodes = [[], [BOS()]] + [[] for i in range(0, size+1)]
        else:
            initial_heap = []
            initial_heap2 = []
            heapq.heappush(initial_heap, (0, BOS()))
            heapq.heappush(initial_heap2, (0, BOS()))
            self.snodes = [initial_heap] + [[] for i in range(0, size+1)]
            self.enodes = [[], initial_heap2] + [[] for i in range(0, size+1)]

        if dst_list is not None:
            self.dst_list = set(dst_list)
        else:
            self.dst_list = None

        if normal_list is not None:
            self.normal_list = set(normal_list)
        else:
            self.normal_list = None

    def add(self, idx, src, dst):
        if self.nbest is None:
            self.add_viterbi(idx, src, dst)
        else:
            self.add_nbest(idx, src, dst)


    def forward(self):
        old_p = self.p
        self.p += 1
        while self.p < len(self.enodes) and not self.enodes[self.p]:
            self.p += 1

        if self.p == len(self.enodes):
            return False

        return self.p - old_p

    def add_viterbi(self, idx, src, dst):
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

    def add_nbest(self, idx, src, dst):
        min_cost = 214748364
        min_path = ()
        min_surface = ()
        min_dst = ''
        if self.p >= len(self.enodes) or len(self.enodes[self.p]) == 0:
            return False

        for i, (m, enode) in enumerate(self.enodes[self.p]):
            if i >= self.nbest:
                break
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
            heapq.heappush(self.snodes[self.p], (min_cost, node))
            heapq.heappush(self.enodes[self.p + len(src)], (min_cost, node))
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
        if self.nbest is None:
            return self.end_viterbi()
        else:
            return self.end_nbest()

    def end_viterbi(self):
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
            print(enode.min_surface)

            cost = enode.min_cost

            if cost < min_cost:
                min_cost = cost
                min_path = enode.min_path
                min_surface = enode.min_surface

        return min_cost, min_path, min_surface

    def end_nbest(self):
        min_cost = 214748364700
        min_path = ()
        min_surface = ()
        res = []
        if self.p >= len(self.enodes) or not self.enodes[self.p]:
            return []

        for i, (m, enode) in enumerate(self.enodes[self.p]):
            if i >= self.nbest:
                break

            if self.dst and enode.dst_str != self.dst:
                continue
            if self.normal_list and enode.dst_str not in self.normal_list:
                continue

            cost = enode.min_cost

            heapq.heappush(res, (cost, enode))

        res = [(r[0], r[1].min_path, r[1].min_surface) for r in res[:self.nbest]]
        return res

    """
    def draw_lattice(self, fn):
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
    """


