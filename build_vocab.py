import dartsclone
import os
import itertools
import argparse
from collections import defaultdict

def build_vocab_splited(fn, outputdir):
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    else:
        print('directory already exists')
        return False

    with open(fn, 'r') as f:
        lines = [line.split('\t') for line in f.read().split("\n") if line != '']
    src_vocab = set()
    pair_vocab = set()

    for line in lines:
        src = set([l if l != 'null' else '' for l in line[0].split()])
        dst = set([l if l != 'null' else '' for l in line[1].split()])
        for s, d in zip(src, dst):
            src_vocab.add(s)
            pair_vocab.add((s, d))
    
    pair_vocab = sorted(list(pair_vocab))
    src_vocab = sorted(list(src_vocab))
    offset = []
    src_value = []
    idx = 0
    for i, word in enumerate(src_vocab):
        s_idx = idx
        while idx < len(pair_vocab) and pair_vocab[idx][0] == word:
            idx += 1
        offset.append(' '.join([str(i), str(s_idx), str(idx)]))
        src_value.append(i)

    costs = []
    for p in pair_vocab:
        if p[0] == p[1]:
            cost = 0
        else:
            cost = len(p[0]) + len(p[1])
        costs.append(p[0] + '/' + p[1] + '\t' + str(cost))

    print(src_vocab[:10])
    print(pair_vocab[:10])
    src_vocab = [s.encode('utf-8') for s in src_vocab]
    src = dartsclone.DoubleArray()
    src.build(src_vocab, values=src_value)
    src.save(os.path.join(outputdir, 'src.dic'))

    src_vocab = [s.decode('utf-8') for s in src_vocab]
    print(src_vocab[:10])
    src_vocab = [str(v) + '\t' + s for s, v in zip(src_vocab, src_value)]

    with open(os.path.join(outputdir, 'src_vocab.txt'), 'w') as f:
        f.write('\n'.join(src_vocab))

    with open(os.path.join(outputdir, 'pair_vocab.txt'), 'w') as f:
        f.write('\n'.join(costs))

    with open(os.path.join(outputdir, 'offset.txt'), 'w') as f:
        f.write('\n'.join(offset))

def build_vocab(fn, window, outputdir):
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    with open(fn, 'r') as f:
        lines = [line.split('\t') for line in f.read().split("\n") if line != '']

    src_vocab = set()
    pair_vocab = set()
    for line in lines:
        assert len(line) >= 2, '２列以上にしてください'
        src = line[0]
        dst = line[1]
        s = []
        d = []

        for idx in range(len(src)):
            for w in range(window):
                if idx + w + 1 > len(src):
                    break
                s.append(src[idx:idx+w+1])

        for idx in range(len(dst)):
            for w in range(window):
                if idx + w + 1 > len(dst):
                    break
                d.append(dst[idx:idx+w+1])

        pair_vocab |= set(itertools.product(s, d+['']))
        #pair_vocab |= set(itertools.product(s, d))
        src_vocab |= set(s)

    pair_vocab = sorted(list(pair_vocab))
    src_vocab = sorted(list(src_vocab))
    offset = []
    src_value = []
    idx = 0
    for i, word in enumerate(src_vocab):
        s_idx = idx
        while idx < len(pair_vocab) and pair_vocab[idx][0] == word:
            idx += 1
        offset.append(' '.join([str(i), str(s_idx), str(idx)]))
        src_value.append(i)

    costs = []
    for p in pair_vocab:
        if p[0] == p[1]:
            cost = 0
        else:
            cost = len(p[0]) + len(p[1])
        costs.append(p[0] + '/' + p[1] + '\t' + str(cost))

    src_vocab = [s.encode('utf-8') for s in src_vocab]
    src = dartsclone.DoubleArray()
    src.build(src_vocab, values=src_value)
    src.save(os.path.join(outputdir, 'src.dic'))

    src_vocab = [s.decode('utf-8') for s in src_vocab]
    src_vocab = [str(v) + '\t' + s for s, v in zip(src_vocab, src_value)]

    with open(os.path.join(outputdir, 'src_vocab.txt'), 'w') as f:
        f.write('\n'.join(src_vocab))

    with open(os.path.join(outputdir, 'pair_vocab.txt'), 'w') as f:
        f.write('\n'.join(costs))

    with open(os.path.join(outputdir, 'offset.txt'), 'w') as f:
        f.write('\n'.join(offset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build vocaburaly files')
    parser.add_argument('--input', type=str, help="specify source file")
    parser.add_argument('--len', type=int, help="specify subword length")
    parser.add_argument('--output', type=str, help="specify output directory")
    parser.add_argument('--splited', action='store_true', help="specify output directory")
    args = parser.parse_args()
    if args.splited:
        build_vocab_splited(args.input, args.output)
    else:
        build_vocab(args.input, args.len, args.output)

