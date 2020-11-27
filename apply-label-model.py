import re
import os
import sys
import time
import glob
import torch
import pandas as pd
import snorkel
import collections
import numpy as np
import argparse
import pickle
import itertools
from snorkel.labeling import LabelModel
from trove.analysis.metrics import split_by_seq_len
from trove.data.dataloaders.contexts import Span


def timeit(f):
    """
    Decorator for timing function calls
    :param f:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'{f.__name__} took: {te - ts:2.4f} sec')
        return result

    return timed


def convert_label_matrix(L):
    # abstain is -1
    # negative is 0
    L = L.toarray().copy()
    L[L == 0] = -1
    L[L == 2] = 0
    return L


def dump_entity_spans(outfpath,
                      y_word_probas,
                      sentences,
                      b=0.5,
                      stopwords=None):
    """
    Given per-word probabilities, generate all
    entity spans (assumes IO tagging)

    """
    stopwords = stopwords if stopwords else None

    snapshot = []
    seq_lens = [len(s) for s in sentences]
    y_sents = split_by_seq_len(y_word_probas[..., 1], seq_lens)

    for sent, y_proba in zip(sentences, y_sents):

        y_pred = [1 if p > b else 0 for p in y_proba]

        curr = []
        spans = []
        for i, (word, y, char_start) in enumerate(
                zip(sent.words, y_pred, sent.abs_char_offsets)):
            if y == 1:
                curr.append((word, char_start))
            elif y == 0 and curr:
                curr = [(w, ch) for w, ch in curr if len(w.strip()) != 0]
                spans.append(list(zip(*curr)))
                curr = []
        if curr:
            curr = [(w, ch) for w, ch in curr if len(w.strip()) != 0]
            spans.append(list(zip(*curr)))

        # initalize spans
        unique_spans = set()
        for s in spans:
            if not s:
                continue

            toks, offsets = s
            start = offsets[0] - sent.abs_char_offsets[0]
            end = offsets[-1] + len(toks[-1]) - sent.abs_char_offsets[0]
            span = Span(char_start=start, char_end=end - 1, sentence=sent)
            if len(span.text) <= 1 or span.text in stopwords:
                continue
            unique_spans.add(span)

        # export rows
        for span in unique_spans:
            norm_str = re.sub('\s{2,}', ' ', span.text.strip())
            norm_str = norm_str.replace('\n', '')
            snapshot.append(
                [span.sentence.document.name, norm_str, span.abs_char_start,
                 span.abs_char_end])
            snapshot[-1] = list(map(str, snapshot[-1]))

    # write to TSV
    with open(outfpath, 'w') as fp:
        for row in snapshot:
            fp.write('\t'.join(row) + '\n')

@timeit
def main(args):

    # -------------------------------------------------------------------------
    # Load Dataset and L Matrices
    # -------------------------------------------------------------------------
    X_sents = pickle.load(open(f'{args.indir}/X_sents', 'rb'))
    X_seq_lens = pickle.load(open(f'{args.indir}/X_seq_lens', 'rb'))
    X_doc_seq_lens = pickle.load(open(f'{args.indir}/X_doc_seq_lens', 'rb'))
    L_words = pickle.load(open(f'{args.indir}/L_words', 'rb'))
    X_words = pickle.load(open(f'{args.indir}/X_words', 'rb'))
    Y_words = pickle.load(open(f'{args.indir}/Y_words', 'rb'))

    for i in range(len(Y_words)):
        freq = collections.Counter()
        for y in Y_words[i]:
            freq[y] += 1
        print(i, freq)

    for i in range(len(L_words)):
        size = L_words[i].shape if L_words[i] is not None else None
        print(f'i={i} {size}')
    print(f'train={args.train} dev={args.dev} test={args.test}')

    # transform label matrices to Snorkel v9.3+ format
    if args.label_fmt == 'snorkel':
        L_words = [
            convert_label_matrix(L_words[0]),
            convert_label_matrix(L_words[1]),
            convert_label_matrix(L_words[2]),
            convert_label_matrix(L_words[3]) \
                if L_words[3] is not None else None
        ]

        Y_words = [
            np.array([0 if y == 2 else 1 for y in Y_words[0]]),
            np.array([0 if y == 2 else 1 for y in Y_words[1]]),
            np.array([0 if y == 2 else 1 for y in Y_words[2]]),
            np.array([])
        ]
        print("Coverted to Snorkel 9.3+ label matrix format")

    # -------------------------------------------------------------------------
    # Load Best Model
    # -------------------------------------------------------------------------
    model = LabelModel(cardinality=2, verbose=True)
    model.load(args.model)
    print(f"Loaded model from {args.model}")

    # -------------------------------------------------------------------------
    # Predict Labels
    # -------------------------------------------------------------------------
    y_proba = model.predict_proba(L_words[args.split])
    sentences = list(
        itertools.chain.from_iterable(
            [doc.sentences for doc in X_sents[args.split]]
        )
    )

    # -------------------------------------------------------------------------
    # Dump Entity Spans
    # -------------------------------------------------------------------------
    dump_entity_spans(args.outdir, y_proba, sentences)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--split", type=int, default=None)
    args = parser.parse_args()

    print(f'PyTorch v{torch.__version__}')
    print(f'Snorkel v{snorkel.__version__}')
    print(args)
    main(args)


