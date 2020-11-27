#
# Tools for loading NER datasets
#
import os
import re
import glob
import random
import numpy as np
import collections
from trove.data.dataloaders.dataloaders import dataloader


###############################################################################
#
# Sampling
#
###############################################################################

def reservoir_sampling(iterable, n, seed=1234):
    """Standard reservoir sampling of a Python iterable."""
    np.random.seed(seed)
    i = 0 
    pool = []
    for item in iterable:
        if len(pool) < n:
            pool.append(item)
        else:
            i += 1
            k = random.randint(0, i)
            if k < n:
                pool[k] = item
    return pool


def load_unlabeled_sample(fpath, num_samples, seed=1234, max_docs=100000):
    """
    Reservoir sample JSON documents. If `seed` and `max_docs` are fixed, 
    then this returns a deterministic subsample of the docs at `fpath`.
    
    """
    filelist = glob.glob(f"{fpath}/*") if os.path.isdir(fpath) else [fpath]
    assert len(filelist) > 0
    
    sample = reservoir_sampling(dataloader(filelist), max_docs, seed)
    return sample[0:num_samples]


###############################################################################
#
# Term Frequency
#
###############################################################################

def ngrams(seq, max_ngrams=4):
    for i in range(0, len(seq)):
        for j in range(i + 1, min(i + max_ngrams + 1, len(seq))):
            term = tuple(seq[i:j]) if ' '.join(seq[i:j]).isupper() else \
                tuple(map(lambda x: x.lower(), seq[i:j]))
            yield term


def ngram_idf(sentences, max_ngrams=4):
    doc_freq = {}
    for s in sentences:
        if s.document.name not in doc_freq:
            doc_freq[s.document.name] = {n: collections.Counter() \
                                         for n in range(1, max_ngrams + 1)}
        words = [w for w in s.words if w.strip()]
        for tokens in set(ngrams(words)):
            term = ' '.join(tokens)
            doc_freq[s.document.name][len(tokens)][term] = 1

    freq = {n: collections.Counter() for n in range(1, max_ngrams + 1)}
    for name in doc_freq:
        for n in doc_freq[name]:
            for term in doc_freq[name][n]:
                freq[n][term] += 1

    for n in freq:
        freq[n] = {term: np.log10(len(doc_freq) / freq[n][term]) \
                   for term in freq[n]}

    return dict(freq)


def get_dict_coverage(term_weights, dictionaries):
    scores = collections.defaultdict(float)
    for name in dictionaries:
        for term in term_weights:
            if term in dictionaries[name]:
                scores[name] += term_weights[term]
    return dict(scores)


def score_umls_ontologies(sentences, ontologies, concepts=None, max_ngrams=4):
    # compute term weights (IDF)
    idf = ngram_idf(sentences, max_ngrams=max_ngrams)

    # compute weighted coverage
    weights = collections.defaultdict(float)
    for ngram in idf:
        coverage = get_dict_coverage(idf[ngram], ontologies)
        for name in coverage:
            weights[name] += coverage[name]

    # restrict to some set of concepts/semantic types
    if concepts:
        sab2sty = collections.defaultdict(set)
        for (sab, sty) in weights:
            sab2sty[sab].add(sty)
        rm_sab = []
        for sab in sab2sty:
            if not sab2sty[sab].intersection(concepts):
                rm_sab.append(sab)

        rm = []
        for (sab, sty) in weights:
            if sab in rm_sab:
                rm.append((sab, sty))

        print(f'Removed {len(rm_sab)} source vocabs, '
              f'{len(rm)} SAB/STY dictionaries')
        for key in rm:
            del weights[key]

    # compute score by SAB (source vocabulary / ontology name)
    scores = collections.defaultdict(float)
    for (sab, sty) in weights:
        scores[sab] += weights[(sab, sty)]

    return scores
