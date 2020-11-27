import torch
import itertools
import numpy as np


class ConllDataset(object):

    def __init__(self, fpath):
        # load sequences and tags
        self.xs, self.ys = ConllDataset.load_file(fpath)
        self.vocab = set(itertools.chain.from_iterable(self.xs))
        self.tags = set(itertools.chain.from_iterable(self.ys))

    @staticmethod
    def load_file(fpath):
        entries = open(fpath, 'r').read().strip().split("\n\n")
        words, tags = [], []
        for entry in entries:
            x = [line.split()[0] for line in entry.splitlines()]
            y = [line.split()[-1] for line in entry.splitlines()]
            words.append(np.array(x))
            tags.append(np.array(y))
        return words, tags


class ProbaConllDataset(object):

    def __init__(self, fpath):
        # load sequences and tags
        self.xs, self.ys, self.masks = ProbaConllDataset.load_file(fpath)
        self.vocab = set(itertools.chain.from_iterable(self.xs))
        self.tags = set(range(0,self.ys[0].shape[1]))

    @staticmethod
    def load_file(fpath):
        entries = open(fpath, 'r').read().strip().split("\n\n")
        words, tags, masks = [], [], []
        for i,entry in enumerate(entries):
            x, y, m = [], [], []
            for line in entry.splitlines():
                line_split = line.split()
                x.append(line_split[0])
                y.append(line_split[1:-1])
                m.append(line_split[-1])

            m = np.array(m).astype(np.bool)
            if sum(m) == 0: # if sequence is completely masked
                continue
            words.append(np.array(x))
            tags.append(np.array(y).astype(np.float32))
            masks.append(m)
        return words, tags, masks


class ProbaConllDataset2(object):

    def __init__(self, fpath):
        # load sequences and tags
        self.xs, self.ys, self.masks = ProbaConllDataset.load_file(fpath)
        self.vocab = set(itertools.chain.from_iterable(self.xs))
        self.tags = set(range(0,self.ys[0].shape[1]))

    @staticmethod
    def load_file(fpath):
        entries = open(fpath, 'r').read().strip().split("\n\n")
        words, tags, masks = [], [], []
        for i,entry in enumerate(entries):
            x, y, m = [], [], []
            for line in entry.splitlines():
                line_split = line.split()
                x.append(line_split[0])
                y.append(line_split[1:-1])
                m.append(line_split[-1])

            m = np.array(m).astype(np.bool)
            if sum(m) == 0: # if sequence is completely masked
                continue
            words.append(np.array(x))
            tags.append(np.array(y).astype(np.float32))
            masks.append(m)
        return words, tags, masks


class SequenceDataset(object):
    """
    Wrapper class for creating indexed sequences.
    """
    def __init__(self, xs, ys, masks, word_to_idx, tag_to_idx):
        self.xs = xs
        self.ys = ys
        self.masks = masks
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.is_proba = len(ys[0].shape) != 1

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        unk = self.word_to_idx['[UNK]']
        x = [self.word_to_idx[w] if w in self.word_to_idx else unk \
             for w in self.xs[item]]
        y = [self.tag_to_idx[i] for i in self.ys[item]] if not self.is_proba \
            else self.ys[item]
        mask = [1] * len(x) if not self.masks else self.masks[item]
        f = np.array
        return f(x), f(y), f(mask), len(x)


class BertSequenceDataset(object):
    """
    Given a corpus tokenized into sentences and words, apply the following for
    use with BERT:
    - apply BERT tokenizer to transform into word pieces
    - enforce max_length constraint
    - add the special BERT [CLS] and [SEP] tokens

    """
    def __init___(self, xs, ys, masks, tokenizer, tag_to_idx, max_length=512):
        self.xs = xs
        self.ys = ys
        self.masks = masks
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx
        self.is_proba = len(ys[0].shape) != 1
        self.max_length = max_length

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        pass


class WordPieceSequenceDataset(object):
    """
    Sequence corpus where word tokens are divided using
    some sub-word scheme (Word Piece, BPE, etc)
    """
    def __init___(self, xs, ys, masks, tokenizer, tag_to_idx, max_length=512):
        pass


def pad_seqs(batch, pad_idx=0):
    f = lambda x: [sample[x] for sample in batch]
    word_indices = f(0)
    tags = f(1)
    mask = f(2)
    seqlens = f(3)

    maxlen = np.array(seqlens).max()
    X, Y, M = [], [], []
    for sample in batch:
        x, y, m, _ = sample
        # store data type so that y is the proper torch tensor type
        dtype = type(y[0])
        pad_len = maxlen - len(x)
        # pad to fixed max length
        x_pad = np.array([pad_idx] * pad_len)
        m_pad = np.array([0] * pad_len)
        # y requires different padding if using probabilistic labels
        y_pad = np.zeros((pad_len, y.shape[1])) \
            if len(y.shape) == 2 else np.zeros((pad_len,))
        X.append(np.concatenate((x, x_pad)))
        Y.append(np.concatenate((y, y_pad)))
        M.append(np.concatenate((m, m_pad)))

    Y = torch.FloatTensor(Y) if len(Y[0].shape) == 2 else torch.LongTensor(Y)
    return torch.LongTensor(X), Y, torch.BoolTensor(M)