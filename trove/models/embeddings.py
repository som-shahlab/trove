import os
import time
import torch
import string
import itertools
import functools
import numpy as np


class Embeddings(object):
    """
    Simple embedding loader. Words with GloVe and FastText text files.
    """
    def __init__(self,
                 fpath:str,
                 fmt:str='text',
                 dim:int=None,
                 normalize:bool=False):
        """

        Parameters
        ----------
        fpath
        fmt
        dim
        normalize
        """
        assert os.path.exists(fpath)
        self.fpath = fpath
        self.dim = dim
        self.fmt = fmt
        # infer dimension
        if not self.dim:
            header = open(self.fpath, "rU").readline().strip().split(' ')
            self.dim = len(header) - 1 if len(header) != 2 else int(header[-1])

        ts = time.time()
        self.vocab, self.vecs = zip(*[(w, vec) for w, vec in self._read()])
        self.vocab = {w: i for i, w in enumerate(self.vocab)}
        self.vecs = np.vstack(self.vecs)

        if normalize:
            norm = np.linalg.norm(self.vecs, axis=1, ord=2)
            self.vecs = (self.vecs.T / norm).T

        print(f"embeddings loaded {time.time() - ts:3.2f} secs")

    def _read(self):
        """

        Returns
        -------

        """
        start = 0 if self.fmt == "text" else 1
        for i, line in enumerate(open(self.fpath, "rU")):
            if i < start:
                continue
            line = line.rstrip().split(' ')
            vec = np.array([float(x) for x in line[1:]])
            if len(vec) != self.dim:
                continue
            yield (line[0], vec)


def match_term(word,
               pretrained_vocab,
               normalize=True):
    """Match word (or normalized form) to terms in the provided
    pre-trained dictionary.

    Parameters
    ----------
    word
    pretrained_vocab
    normalize

    Returns
    -------

    """
    surface_forms = [
        word, word.lower(), word.strip(string.punctuation)
    ] if normalize else [word]
    for w in surface_forms:
        if w in pretrained_vocab:
            return True
    return False


def build_vocab(datasets,
                pretrained_vocab=None,
                specials=None,
                normalize_vocab=True):
    """Build a vocabulary set for a collection of data sets.

    Parameters
    ----------
    datasets
    pretrained_vocab
    specials

    Returns
    -------

    """
    match = functools.partial(
        match_term,
        pretrained_vocab=pretrained_vocab,
        normalize=normalize_vocab
    )

    # initialize training vocabulary
    train, dev, test = datasets
    specials = ['[PAD]', '[UNK]'] if not specials else specials
    vocab = {word:i for i,word in enumerate(specials)}
    vocab.update({word:i + len(vocab) for i,word in enumerate(train.vocab)})
    if not pretrained_vocab:
        return vocab

    # map dev/test to pre-trained vocab
    valid_vocab = set(itertools.chain.from_iterable([dev.vocab, test.vocab]))
    valid_vocab = {w for w in valid_vocab if w not in vocab}
    for word in valid_vocab:
        if match(word):
            vocab[word] = len(vocab)

    assert len(vocab) == len(set(vocab.values()))

    # vocab mapping statistics
    n_train_oov = len([w for w in train.vocab if not match(w)])
    n_valid_oov = len([w for w in valid_vocab if not match(w)])
    print('-' * 80)
    print(f'OOV (train)    {n_train_oov/len(train.vocab)*100:2.1f}% ' +
          f'({n_train_oov}/{len(train.vocab)})')
    print(f'OOV (dev+test) {n_valid_oov/len(valid_vocab)*100:2.1f}% ' +
          f'({n_valid_oov}/{len(valid_vocab)})')

    # coverage of pre-trained embeddings
    V = set(itertools.chain.from_iterable([x.vocab for x in datasets]))
    n = len([w for w in V if match(w)])
    N = len(V) - len(specials)
    print(f'Matched {n/N*100:2.1f}% ({n}/{N})')
    print(f'Vocab: {len(vocab)}')
    print(f'{n/len(vocab)*100:2.2f}% ({n}/{len(vocab)}) pretrained init')
    print('-' * 80)
    return vocab


def init_pretrained_embs(embs,
                         vocab,
                         specials = None,
                         seed = 0):
    """Initialize pre-trained word embedding weight matrix.

    Parameters
    ----------
    embs
    vocab
    specials

    Returns
    -------

    """
    torch.manual_seed(seed)

    specials = ['[PAD]', '[UNK]'] if not specials else specials
    W = torch.nn.init.xavier_normal_(
        torch.empty(len(vocab), embs.vecs.shape[1])
    )
    for word,i in vocab.items():
        if word in specials:
            continue
        for w in [word, word.lower(), word.strip(string.punctuation)]:
            if w in embs.vocab:
                idx = embs.vocab[w]
                W[i] = torch.FloatTensor(embs.vecs[idx])
                break
    return W