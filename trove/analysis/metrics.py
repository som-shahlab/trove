import itertools
import numpy as np
import seqeval.metrics
from trove.data.dataloaders.ner import entity_tag
from typing import List, Set, Dict, Tuple, Pattern, Match, Iterable


def split_by_seq_len(X, X_lens) -> np.ndarray:
    """Given a matrix X of M elements, partition into N variable length
    sequences where [xi, ..., xN] lengths are defined by X_lens[i].

    This is used to partition a stacked matrix of words back into sentences.

    Parameters
    ----------
    X
    X_lens

    Returns
    -------

    """
    splits = [np.sum(X_lens[0:i]) for i in range(1, X_lens.shape[0])]
    return np.split(X, splits)


def convert_tag_fmt(
        seq: List[str],
        etype: str,
        tag_fmt: str = 'IOB') -> List[str]:
    """Convert between tagging schemes. This is a lossy conversion
    when converting to IO, i.e., mapping {IOB, IOBES} -> IO
    drops information on adjacent entities.

    IOB -> O B I I B I O
    IO  -> O I I I I I O
    IOB -> O B I I I I O

    Parameters
    ----------
    seq
    etype
    tag_fmt

    Returns
    -------

    """
    # TODO: Only works for IO -> {IOB, IOBES}
    assert set(seq).issubset(set('IO'))
    # divide into contiguous chunks
    chunks = [list(g) for _, g in itertools.groupby(seq)]
    # remap to new tagging scheme
    seq = list(itertools.chain.from_iterable(
        [tags if 'O' in tags else entity_tag(len(tags), tag_fmt)
         for tags in chunks]
    ))
    return [t if t == 'O' else f'{t}-{etype}' for t in seq]


def tokens_to_sequences(y_gold,
                        y_pred,
                        seq_lens,
                        idx2tag=None,
                        tag_fmt=None):
    """Convert token labels to sentences for sequence model evaluation.

    Parameters
    ----------
    y_gold
    y_pred
    seq_lens
    idx2tag
    tag_fmt

    Returns
    -------

    """
    idx2tag = {1: 'I', 0: 'O'} if not idx2tag else idx2tag
    y_gold_seqs = []
    for s in split_by_seq_len(y_gold, seq_lens):
        y = [idx2tag[i] for i in s]
        if tag_fmt is not None:
            y_hat = convert_tag_fmt(y, etype='ENTITY', tag_fmt='IOB')
        else:
            y_hat = y
        y_gold_seqs.append(y_hat)

    y_pred_seqs = []
    for s in split_by_seq_len(y_pred, seq_lens):
        # HACK -- sometimes -1 labels make it into evaluation due to Snorkel
        # label model. Just treat these as 'O'
        y = [idx2tag[i] if i in idx2tag else 'O' for i in s]
        if tag_fmt is not None:
            y_hat = convert_tag_fmt(y, etype='ENTITY', tag_fmt='IOB')
        else:
            y_hat = y
        y_pred_seqs.append(y_hat)

    return y_gold_seqs, y_pred_seqs


def score_sequences(y_true: List[List[int]],
                    y_pred: List[List[int]],
                    metrics: Set[str] = None) -> Dict[str, float]:
    """
    Sequence model evaluation using seqeval
    https://github.com/chakki-works/seqeval

    Parameters
    ----------
    y_gold
    y_pred

    Returns
    -------

    """
    scorers = {
        'accuracy': seqeval.metrics.accuracy_score,
        'precision': seqeval.metrics.precision_score,
        'recall': seqeval.metrics.recall_score,
        'f1': seqeval.metrics.f1_score
    }
    metrics = metrics if metrics is not None else scorers
    try:
        return {name: scorers[name](y_true, y_pred) for name in metrics}
    except:
        return {name: 0.0 for name in metrics}


