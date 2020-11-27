import re
import os
import sys
import time
import glob
import torch
import pandas as pd
import snorkel
import numpy as np
import argparse
import pickle
import itertools
from snorkel.labeling import LabelModel
from trove.analysis.metrics import split_by_seq_len
from trove.data.dataloaders.contexts import Span
from pytorch_pretrained_bert import BertTokenizer
from trove.data.dataloaders.ner import NerDocumentDataset, load_json_dataset


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


def split_span(span, split_on='\n'):


    S = []
    char_start = span.char_start
    subspans = re.split(r'''(\n)''', span.text)
    for s in subspans:
        if s == '\n':
            char_start += len(s)
            continue
        S.append((char_start, char_start + len(s)))
        char_start += len(s)

    spans = []
    for i, j in S:
        s = Span(char_start=i, char_end=j, sentence=span.sentence)
        print("+", s)
        spans.append(s)
    return span

def dump_entity_spans(outfpath,
                      y_word_probas,
                      sentences,
                      b=0.5,
                      min_length = 0,
                      stopwords=None):
    """
    Given per-word probabilities, generate all
    entity spans (assumes IO tagging)

    """
    stopwords = stopwords if stopwords else {}

    snapshot = []
    seq_lens = np.array([len(s.words) for s in sentences])
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

            # HACK filter drugs
            if re.search(r'''^(([0-9][.,]*[0-9]*) (mg)|mg |["])''', span.text):
                continue

            if len(span.text) <= min_length or span.text in stopwords:
                continue

            # HACK - an artifact of IO tagging, sometimes we concatenate
            # spans of separate entities. This almost exclusively happens
            # when spans are split by 1 newline, so it's easy to fix by
            # splitting spans into sub-spans by newline

            # newlines = [j for j in sent.newlines if j >= start and j <= end]
            # if newlines:
            #     #split_span(span, newlines)
            #     print("++", span)
            #     subspans = [0] + [i + 1 for i in newlines] + [len(sent) + 1]
            #     for i in range(len(subspans) - 1):
            #         subspan = Span(char_start=subspans[i], char_end=subspans[i+1] - 1, sentence=sent)
            #         print('-', subspan)

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


def get_mask(L):
    mask = []
    for i,row in enumerate(L):
        if list(row).count(-1) == L.shape[-1]:
            mask.append(True)
        else:
            mask.append(False)
    return mask



@timeit
def main(args):

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                              do_lower_case=False)
    # load dataset
    tag_fmt = 'BIO'

    proba_dataset = {
        'train': load_json_dataset(args.train, tokenizer, tag_fmt=tag_fmt),
        'dev'  : load_json_dataset(args.dev, tokenizer, tag_fmt=tag_fmt),
        'test' : load_json_dataset(args.test, tokenizer, tag_fmt=tag_fmt)
    }

    tagged = [
        [(words[1:-1], tags[1:-1]) for words, x, is_heads, tags, y, seqlens in
         proba_dataset[name]]
        for name in proba_dataset
    ]

    # -------------------------------------------------------------------------
    # Load Best Model
    # -------------------------------------------------------------------------
    model = LabelModel(cardinality=2, verbose=True)
    model.load(args.model)
    print(f"Loaded model from {args.model}")

    # -------------------------------------------------------------------------
    # Load Dataset and L Matrices
    # -------------------------------------------------------------------------
    X_sents = pickle.load(open(f'{args.indir}/X_sents', 'rb'))
    X_seq_lens = pickle.load(open(f'{args.indir}/X_seq_lens', 'rb'))
    X_doc_seq_lens = pickle.load(open(f'{args.indir}/X_doc_seq_lens', 'rb'))
    L_words = pickle.load(open(f'{args.indir}/L_words', 'rb'))
    X_words = pickle.load(open(f'{args.indir}/X_words', 'rb'))
    Y_words = pickle.load(open(f'{args.indir}/Y_words', 'rb'))

    for i in range(len(L_words)):
        size = L_words[i].shape if L_words[i] is not None else None
        print(f'i={i} {size}')

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
    # Label model inference
    # -------------------------------------------------------------------------
    use_unlabeled = False

    sentences = [
        split_by_seq_len(X_words[i], X_seq_lens[i])
        for i in (range(len(X_words)) if use_unlabeled else range(3))
    ]

    probas = [
        model.predict_proba(L_words[i])
        for i in (range(len(X_words)) if use_unlabeled else range(3))
    ]

    gold_seqs = [
        split_by_seq_len(Y_words[i], X_seq_lens[i])
        for i in (range(len(Y_words)) if use_unlabeled else range(3))
    ]

    proba_seqs = [
        split_by_seq_len(probas[i], X_seq_lens[i])
        for i in (range(len(X_words)) if use_unlabeled else range(3))
    ]

    masks = [
        get_mask(L_words[i])
        for i in (range(len(X_words)) if use_unlabeled else range(3))
    ]

    mask_seqs = [
        split_by_seq_len(masks[i], X_seq_lens[i])
        for i in (range(len(X_words)) if use_unlabeled else range(3))
    ]

    # -------------------------------------------------------------------------
    # Export Proba CONLL
    # -------------------------------------------------------------------------

    splits = ['train', 'dev', 'test']
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                              do_lower_case=False)

    n_errs = 0
    N = 0
    n_b_fixes = 0

    prior = 'balance'

    for split in range(0, 3):

        data = []
        # gold tags
        words, tags = zip(*tagged[split])

        for xs, ys, ps, ms, ws, ts in zip(
                sentences[split], gold_seqs[split], proba_seqs[split],
                mask_seqs[split], words, tags
        ):

            sent = []
            for x, y, p, m, w, t in zip(xs, ys, ps, ms, ws, ts):
                toks = tokenizer.tokenize(x)
                tokens = []
                curr = []
                for i in range(0, len(toks)):
                    if toks[i][:2] != '##':
                        if curr:
                            tokens.append(curr)
                        curr = [toks[i]]
                    elif toks[i][:2] == '##':
                        curr.append(toks[i][2:])

                if curr:
                    tokens.append(curr)

                tokens = [''.join(t) for t in tokens]

                # confirm that tokenization works
                if ''.join(x) != ''.join(tokens):
                    print("ERR")
                    print(x)
                    print(toks)
                    print(tokens)
                    print("======")

                p = p if (not m or prior != 'uniform') else [1.0 / len(
                    p)] * len(p)

                # HACK for B-tokens
                tags = [t[0]] * len(tokens)
                if t[0] == 'B' and len(tokens) > 1:
                    n_b_fixes += 1
                    tags[1:] = 'I'

                for tok, tag in zip(tokens, tags):
                    sent.append(
                        f'{tok}\t{p[0]:1.5f}\t{p[1]:1.5f}\t{int(not m)}\t{tag}')

            data.append('\n'.join(sent) + '\n')

        fpath = f'{args.outdir}/{args.etype}.{splits[split]}.prior_{prior}.proba.conll.tsv'
        print(fpath)
        with open(fpath, 'w') as fp:
            fp.write('\n'.join(data))



    print(n_errs, N)
    print("DONE")



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--indir", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    #parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--label_fmt", type=str, default='snorkel')
    parser.add_argument("--etype", type=str, default="entity")

    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--dev", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)

    args = parser.parse_args()

    print(f'PyTorch v{torch.__version__}')
    print(f'Snorkel v{snorkel.__version__}')
    print(args)
    main(args)


# python label-model-proba-conll.py \
# --model /Users/fries/Desktop/covid-gold-docs/label_models/drug/123/model.n2331042.1596478182.pkl \
# --indir /Users/fries/Desktop/covid-gold-docs/label_matrices/drug/123/ \
# --outdir /Users/fries/Desktop/covid-gold-docs/proba_conll/drug/123/ \
# --etype drug


# python label-model-proba-conll.py \
# --model /Users/fries/Desktop/covid-gold-docs/label_models/diso_sign_symp/123//model.n2331042.1596481792.pkl \
# --train /Users/fries/Desktop/covid-train/final/train.shc-covid.json \
# --dev /Users/fries/Desktop/NER-Datasets-ALL/ehr/shareclef/experiments/dev.shareclef.disorder.json \
# --test /Users/fries/Desktop/covid-gold-docs/gold/diso_sign_symp/covid-shc.diso_sign_symp.gold.json \
# --indir /Users/fries/Desktop/covid-gold-docs/label_matrices/diso_sign_symp/123/ \
# --outdir /Users/fries/Desktop/covid-gold-docs/proba_conll/diso_sign_symp/123/ \
# --etype diso_sign_symp

#

# python label-model-proba-conll.py \
# --model /Users/fries/Desktop/covid-gold-docs/label_models/diso_sign_symp/123//model.n2331042.1596500481.pkl \
# --train /Users/fries/Desktop/covid-train/final/train.shc-covid.json \
# --dev /Users/fries/Desktop/NER-Datasets-ALL/ehr/shareclef/experiments/dev.shareclef.disorder.json \
# --test /Users/fries/Desktop/covid-gold-docs/gold/diso_sign_symp/covid-shc.diso_sign_symp.gold.json \
# --indir /Users/fries/Desktop/covid-gold-docs/label_matrices/diso_sign_symp/123/ \
# --outdir /Users/fries/Desktop/covid-gold-docs/proba_conll/diso_sign_symp/123/ \
# --etype diso_sign_symp
