#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Loads and preprocesses the "Wikitext long term dependency
language modeling dataset:
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

Lowercases all the words and splits into sentences, omitting
the ending period. Add sentence start/end tokens.
"""

import os
from multiprocessing import Pool
from vocab import Vocab

SENT_START = "<s>"
SENT_END = "</s>"

def path(part):
    """ Gets the dataset for 'part' being train|test|valid. """
    assert part in ("train", "test", "valid")
    return os.path.join("/home/ykang7/Data/corpus", "tiny_corpus" + ".txt")


class F(object):
    def __init__(self, vocab:list):
        self.vocab = vocab

    def __call__(self, line:str):
        new_line = [SENT_START] + line.split() + [SENT_END]
        return [self.vocab.index(w) for w in new_line]


def load(path):
    """ Loads the wikitext2 data at the given path using
    the given index (maps tokens to indices). Returns
    a list of sentences where each is a list of token
    indices.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    vocab = set(''.join(lines).split()).union({SENT_START, SENT_END})
    vocab = list(sorted(vocab))
    f = F(vocab)
    with Pool(4) as p:
        sentences = p.map(f, lines)

    return Vocab(vocab), sentences


def main():
    print("WikiText2 preprocessing test and dataset statistics")
    index = Vocab()
    for part in ("train", "valid", "test"):
        print("Processing", part)
        sentences = load(path(part), index)
        print("Found", sum(len(s) for s in sentences),
              "tokens in", len(sentences), "sentences")
    print("Found in total", len(index), "tokens")


if __name__ == '__main__':
    main()
