import re
import gzip
import json
import torch
import logging
import itertools
import collections
import pdb
import numpy as np
import gc

from torch.utils import data
from scipy.sparse.csr import csr_matrix
from pytorch_pretrained_bert import BertTokenizer
from .contexts import Document, Sentence, Annotation
from typing import List, Set, Dict, Tuple, Pattern, Match, Iterable, Union, Iterator

#from overrides import overrides

logger = logging.getLogger(__name__)


def load_json_dataset(fpath: str,
                      tokenizer: Union[str, BertTokenizer],
                      tag_fmt: str = 'IO',
                      contiguous_only: bool = False):
    """Load JSON dataset and initialize sequence tagged labels.

    Parameters
    ----------
    fpath
        JSON file path
    tokenizer

    tag_fmt
        token tagging scheme with values in {'IO','IOB', 'IOBES'}
    """
    documents, entities = [], {}
    fopen = gzip.open if fpath.split(".")[-1] == 'gz' else open
    with fopen(fpath, 'rb') as fp:
        for line in fp:
            # initialize context objects
            d = json.loads(line)
            doc = Document(d['name'], [Sentence(**s) for s in d['sentences']])
            documents.append(doc)
            # load entities
            entities[doc.name] = set()
            if 'entities' not in d:
                continue
            for entity in d['entities']:
                del entity['abs_char_start']
                del entity['abs_char_end']
                if 'doc_name' not in entity:
                    entity['doc_name'] = doc.name

                anno = Annotation(**entity)
                if len(anno.span) > 1 and contiguous_only:
                    continue

                entities[doc.name].add(Annotation(**entity))

    return NerDocumentDataset(documents,
                              entities,
                              tag_fmt=tag_fmt,
                              tokenizer=tokenizer)


#################################################################################
#
#  Sequence Tag Creation
#
#################################################################################

def entity_tag(length, tag_fmt="IOB"):
    """
    IO, IOB, or IOBES (equiv. to BILOU) tagging

    :param tokens:
    :param is_heads:
    :param tag_fmt:
    :return:
    """
    tags = ['O'] * length
    tag_fmt = set(tag_fmt)

    if tag_fmt == set("IOB"):
        tags[0] = 'B'
        tags[1:] = len(tags[1:]) * "I"

    elif tag_fmt == set("IOBES") or tag_fmt == set("BILOU"):
        if len(tags) == 1:
            tags[0] = 'S'
        else:
            tags[0] = 'B'
            tags[1:-1] = len(tags[1:-1]) * "I"
            tags[-1:] = "E"

    elif tag_fmt == set("IO"):
        tags = ['I'] * len(tags)
    return tags


def map_sent_entities(document, entities, verbose=True):
    """
    Given (1) a document split into sentences and (2) a list of entities
    defined by absolute char offsets, map each entity to it's parent sentence.

    :param:
    :param:
    :return tuple of sentence index and tag,
    """
    errors = 0
    spans = []
    char_index = [s.abs_char_offsets[0] for s in document.sentences]

    for t in entities:
        position = None
        for i in range(len(char_index) - 1):
            if t.abs_char_start >= char_index[i] and t.abs_char_end <= char_index[i + 1]:
                position = i
                break

        if position == None and t.abs_char_start >= char_index[-1]:
            position = len(char_index) - 1

        if position == None:
            values = (document.name, t.abs_char_start, t.abs_char_end)
            if verbose:
                msg = f"{[t.text]} {t.span} {t.doc_name}"
                logger.warning(f"Cross-sentence mention {msg}")
            errors += 1
            continue
        try:
            shift = document.sentences[position].abs_char_offsets[0]
            span = document.sentences[position].text[t.abs_char_start - shift:t.abs_char_end - shift]
            spans.append((position, t, span))
        except Exception as e:
            logger.error(f'{e}')

    idx = collections.defaultdict(list)
    for i, entity, _ in spans:
        idx[i].append(entity)

    return idx, errors


def retokenize(sent, tokenizer):
    """
    Given a default tokenization, compute absolute character offsets for
    a new tokenization (e.g., BPE). By convention word piece tokens are
    prefixed by ## .

    :param sent:
    :param tokenizer:
    :return:
    """
    toks = [tokenizer.tokenize(w) for w in sent.words]

    text = sent.text
    char_offsets = []
    start = sent.abs_char_offsets[0]

    for i, word in zip(sent.abs_char_offsets, toks):
        offsets = [i]
        for t in word:
            if t[0:2] == '##':
                t = t[2:]

            substr = text[offsets[-1] - start:offsets[-1] + len(t) - start]
            if t != substr:
                offsets[-1] += 3

            offsets.append(offsets[-1] + len(t))
        char_offsets.append(offsets)

    tokens, abs_char_offsets = [], []
    for i, w in zip(char_offsets, toks):
        abs_char_offsets.extend(i[0:-1])
        if w:
            tokens.extend(w)

    return tokens, abs_char_offsets


def tokens_to_tags(sent, entities,
                   tag_fmt='BIO',
                   tokenizer=None,
                   max_seq_len=512):
    """

    :param sent:
    :param entities:
    :param tag_fmt:
    :param tokenizer:
    :param max_seq_len:
    :return:
    """
    errs = 0
    toks, abs_char_offsets = retokenize(sent, tokenizer) if tokenizer \
        else (sent.words, sent.abs_char_offsets)

    if len(toks) > max_seq_len - 2:
        toks = toks[0:max_seq_len - 2]
        abs_char_offsets = abs_char_offsets[0:max_seq_len - 2]

    # use original tokenization to assign token heads
    is_heads = [1 if i in sent.abs_char_offsets else 0 for i in abs_char_offsets]
    tags = ['O'] * len(toks)

    for entity in entities:
        head = entity.span[0]
        if head[0] in abs_char_offsets:
            start = abs_char_offsets.index(head[0])
            end = len(abs_char_offsets)

            for j, offset in enumerate(abs_char_offsets):
                if head[-1] > offset:
                    continue
                end = j
                break

            # tokenization error
            if is_heads[start] == 0:
                errs += 1
                logger.error(f"Tokenization Error: Token is not a head token {entity} {sent.document.name}")
                continue

            tok_len = is_heads[start:end].count(1)

            head_tags = entity_tag(tok_len, tag_fmt=tag_fmt)
            head_tags = [f'{t}-{entity.type}' for t in head_tags]
            io_tags = ['O'] * len(toks[start:end])

            for i in range(len(io_tags)):
                if is_heads[start:end][i] == 1:
                    t = head_tags.pop(0)
                io_tags[i] = t

            tags[start:end] = io_tags

            # Error Checking: do spans match?
            s1 = ''.join([w if w[:2] != '##' else w[2:] for w in toks[start:end]]).lower()
            s2 = re.sub(r'''(\s)+''', '', entity.text).lower()
            if s1 != s2:
                if len(entity.span) == 1:
                    msg = f"{s1} != {s2}"
                    logger.error(f"Span does not match {msg}")
                errs += 1
        else:
            errs += 1
            logger.error(f"Tokenization Error: Token head not found in abs_char_offsets {entity}")

    return (toks, tags, is_heads), errs


def tag_doc(doc, entities, tag2idx, tag_fmt='IO', tokenizer=None,
            max_seq_len=512):
    seqs = []
    # assign entities to their parent sentence
    sent_map, errs = map_sent_entities(doc, entities)
    missing_heads = 0

    for sent in doc.sentences:

        # optionally retokenize sequence when aligning entity spans
        if tokenizer:
            toks, abs_char_offsets = retokenize(sent, tokenizer)
        else:
            toks, abs_char_offsets = sent.words, sent.abs_char_offsets

        # BERT will give an error for longer sequences
        if len(toks) > max_seq_len:
            logger.warning(f'Truncating sentence tokens {len(toks)} to {max_seq_len}')
            # always assume 2 extra tokens for [CLS] and [SEP] special BERT tokens
            toks = toks[0:max_seq_len - 2]
            abs_char_offsets = abs_char_offsets[0:max_seq_len - 2]

        is_heads = [1 if i in sent.abs_char_offsets else 0 for i in abs_char_offsets]
        tags = ['O'] * len(toks)

        # tag entities
        if sent.i in sent_map:
            for entity in sent_map[sent.i]:
                head = entity.span[0]
                if head[0] in abs_char_offsets:
                    start = abs_char_offsets.index(head[0])
                    end = len(abs_char_offsets)

                    for j, offset in enumerate(abs_char_offsets):
                        if head[-1] > offset:
                            continue
                        end = j
                        break

                    # if token is not a head (i.e. a subword entity match)
                    # promote first token to a head
                    if is_heads[start] == 0:
                        is_heads[start] = 1
                        missing_heads += 1

                    tok_len = is_heads[start:end].count(1)

                    head_tags = entity_tag(tok_len, tag_fmt=tag_fmt)
                    head_tags = [f'{t}-{entity.type}' for t in head_tags]
                    io_tags = ['O'] * len(toks[start:end])

                    for i in range(len(io_tags)):
                        if is_heads[start:end][i] == 1:
                            t = head_tags.pop(0)
                        io_tags[i] = t

                    tags[start:end] = io_tags

                    # do spans match?
                    s1 = ''.join([w if w[:2] != '##' else w[2:] for w in toks[start:end]]).lower()
                    s2 = re.sub(r'''(\s)+''', '', entity.text).lower()

                    if s1 != s2:
                        if len(entity.span) == 1:
                            msg = f"{s1} != {s2}"
                            logger.error(f"Span does not match {msg}")
                        errs += 1
                else:
                    errs += 1
                    logger.error(f"Entity token head not found in abs_char_offsets")

        # add BERT special tokens
        ##words = sent.words[0:is_heads.count(1) + 1]
        # if len(words) > 200:
        #    print(words)
        words = ['[CLS]'] + sent.words + ['[SEP]']
        toks = ['[CLS]'] + toks + ['[SEP]']
        tags = ['X'] + tags + ['X']
        is_heads = [1] + is_heads + [1]

        if len(words) != len(tags):
            print('WTF')
            print(words)

        # create BERT inputs
        x = tokenizer.convert_tokens_to_ids(toks)
        y = [tag2idx[t] if h == 1 else tag2idx['X'] for t, h in zip(tags, is_heads)]
        seqlen = len(y)

        # original tags (head words only)
        tags = [t for i, t in enumerate(tags) if is_heads[i] == 1]
        seqs.append((words, x, is_heads, tags, y, seqlen))

    return seqs, errs, missing_heads


#################################################################################
#
#  Datasets
#
#################################################################################


class NerDocumentDataset(object):
    """
    Document + Annotation objects
    entities are defined as abs char offsets per document
    """

    def __init__(self, documents: dict,
                 entities: dict,
                 tag_fmt: str = 'IO',
                 tokenizer=None) -> None:
        """
        Convert Document objects with a corresponding entity set into tagged sequences

        :param documents:
        :param entities:
        :param tag_fmt:
        :param tokenizer:
        """
        self.documents = documents
        self.entities = entities
        self.tag_fmt = tag_fmt
        self.tokenizer = tokenizer
        self.tag2idx = self._get_tag_index(entities, tag_fmt)

        self._init_sequences(documents)

    def _get_tag_index(self, entities, tag_fmt):
        """
        Given a collection of entity types, initialize an integer tag mapping
        e.g., B-Drug I-Drug O

        :param entities:
        :param tag_fmt:
        :return:
        """
        entity_types = {t.type for doc_name in entities for t in entities[doc_name]}
        tags = [t for t in list(tag_fmt) if t != 'O']
        tags = [f'{tag}-{etype}' for tag, etype in itertools.product(tags, entity_types)]
        tags = ['X', 'O', ] + tags
        return {t: i for i, t in enumerate(tags)}

    def __len__(self) -> int:
        return len(self.data)

    def tagged(self, idx):
        """
        Return tagged words
        :return:
        """

        X, _, _, Y, _, _ = self.__getitem__(idx)
        return X[1:-1], Y[1:-1]

    def _init_sequences_v1(self, documents):
        """
        Transform Documents into labeled sequences.

        :param documents:
        :return:
        """
        self.data = []
        self.sentences = []
        num_errors, num_missing_heads, num_entities = 0, 0, 0

        for doc in documents:
            annotations = self.entities[doc.name] if doc.name in self.entities else {}
            num_entities += len(annotations)

            seqs, errs, missing_heads = tag_doc(doc,
                                                entities=annotations,
                                                tag2idx=self.tag2idx,
                                                tag_fmt=self.tag_fmt,
                                                tokenizer=self.tokenizer)
            num_errors += errs
            num_missing_heads += missing_heads
            self.data.extend(seqs)
            self.sentences.extend(doc.sentences)

        assert len(self.data) == len(self.sentences)
        if num_errors or missing_heads:
            msg = f'Errors: Span Alignment: {num_errors}/{num_entities} ({num_errors / num_entities * 100:2.1f}%)'
            msg += f' Head Tokenization: {num_missing_heads}/{num_entities}'
            logger.warning(msg)

        print(f'Tagged Entities: {num_entities - num_errors}')

    def _init_sequences(self, documents):
        """
        Transform Documents into labeled sequences.

        :param documents:
        :return:
        """
        self.data = []
        self.sentences = []
        num_errors, num_missing_heads, num_entities = 0, 0, 0

        for doc in documents:
            self.sentences.extend(doc.sentences)
            annotations = self.entities[doc.name] if doc.name in self.entities else {}
            num_entities += len(annotations)
            # tag sentences
            sent_entities, errs = map_sent_entities(doc, annotations)
            num_errors += errs

            for sentence in doc.sentences:
                entities = sent_entities[sentence.i] if sentence.i in sent_entities else []
                seqs, errs = tokens_to_tags(sentence, entities, self.tag_fmt, tokenizer=self.tokenizer)
                num_errors += errs

                x, y, is_heads = seqs
                if not (len(x) == len(y) == len(is_heads)):
                    print(seqs)

                self.data.append(seqs)

        assert len(self.data) == len(self.sentences)
        if num_errors:
            msg = f'Errors: Span Alignment: {num_errors}/{num_entities} ({num_errors / num_entities * 100:2.1f}%)'
            logger.warning(msg)

        print(f'Tagged Entities: {num_entities - num_errors}')

    def __getitem__(self, idx):

        toks, tags, is_heads = self.data[idx]

        words = [w for w in self.sentences[idx].words if w.strip()]
        words = self.sentences[idx].words
        # original tags (head words only)
        tags = [t for i, t in enumerate(tags) if is_heads[i] == 1]

        if len(words) != len(tags):
            print(len(words), len(tags))
            print(words)
            print(tags)
            print('-' * 50)

        words = ['[CLS]'] + words + ['[SEP]']
        toks = ['[CLS]'] + toks + ['[SEP]']
        tags = ['X'] + tags + ['X']

        X = self.tokenizer.convert_tokens_to_ids(toks)
        Y = [self.tag2idx[t] if h == 1 else self.tag2idx['X'] for t, h in zip(tags, is_heads)]

        return words, X, is_heads, tags, Y, len(Y)


class SequenceLabelingDataset(data.Dataset):
    """
    Sequence Labeled dataset assumes X, Y tagged sequences
    Supports hard and probabalistic y labels

    """

    def __init__(self,
                 X,
                 Y,
                 tag2idx: Dict[str, int],
                 tokenizer='bert-base-cased',
                 max_seq_len: int = 512,
                 M=None):

        self.X = X
        self.Y = Y
        self.M = M

        self.tag2idx = tag2idx
        self.idx2tag = {i: tag for tag, i in tag2idx.items()}
        if type(tokenizer) is str:
            lc = '-cased' not in tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer,
                                                           do_lower_case=lc)
        else:
            self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.X)

    def _get_seqs(self, idx):
        """
        Ignore whitespace characters (e.g., newlines, spaces, tabs)

        :param idx:
        :return:
        """
        X = np.array(self.X[idx])
        Y = self.Y[idx]
        M = np.array(self.M[idx] if self.M else [1] * len(X))

        # transform to dense array
        Y = Y.toarray() if type(Y) is csr_matrix else np.array(Y)
        return X, Y, M

    def __getitem__(self, idx):

        X, Y, M = self._get_seqs(idx)

        # probabalistic labels
        is_proba = True if len(Y.shape) > 1 else False

        # add BERT special tokens
        X = np.concatenate((['[CLS]'], X, ['[SEP]']))
        M = np.concatenate(([1], M, [1]))
        if is_proba:
            pad = np.zeros((1, 2))
            Y = np.concatenate((pad, Y, pad))
        else:
            Y = np.concatenate((['X'], Y, ['X']))
            Y = np.array([self.tag2idx[t] for t in Y])

        x, y = [], []
        is_heads = []

        for i, (word, tag) in enumerate(zip(X, Y)):
            tokens = self.tokenizer.tokenize(word) if word not in ('[CLS]', '[SEP]') else [word]
            tag = tag.reshape(1, -1) if is_proba else [tag]

            pad_len = len(tokens) - 1
            if pad_len != 0:
                pad = np.zeros((pad_len, Y.shape[-1])) if is_proba else [self.tag2idx['X']] * (len(tokens) - 1)
                tag = np.concatenate((tag, pad))

            # preserve existing tokenization
            is_head = [1] + [0] * (len(tokens) - 1)
            # or use default BERT heads (TODO: note this breaks gold tags ATM)
            # is_head = [1 if tok[0:2] != '##' else 0 for tok in tokens]

            is_heads.extend(is_head)

            y.append(tag)
            x.extend(self.tokenizer.convert_tokens_to_ids(tokens))

        y = np.concatenate(y)

        if len(x) > self.max_seq_len:
            x = np.concatenate((x[:self.max_seq_len - 1], x[-1:]))
            y = np.concatenate((y[:self.max_seq_len - 1], y[-1:]))
            is_heads = np.concatenate((is_heads[:self.max_seq_len - 1], is_heads[-1:]))

        # return x, is_heads, y, len(x)

        # X = ['[CLS]'] + self.X[idx] + ['[SEP]']
        # if not is_proba:
        #     #Y = ['X'] + self.Y[idx] + ['X']
        #     pass
        # else:
        #     # probabalistic labels do not have a
        #     Y = self.Y[idx]
        #     Y = Y.toarray() if type(Y) is csr_matrix else Y
        #     Y = ['X'] + (Y.shape[0] * ['NA']) + ['X']

        Y = np.array([self.idx2tag[t] for t in Y])

        return X, x, is_heads, Y, y, len(x)


class ProbaSequenceLabelingDataset(data.Dataset):
    """
    Sequence Labeled dataset assumes X, Y tagged sequecnes
    Supports hard and probabalistic y labels

    """

    def __init__(self,
                 X,
                 Y,
                 tag2idx: Dict[str, int],
                 tokenizer='bert-base-cased',
                 max_seq_len: int = 512,
                 M=None,
                 for_mtl=False):

        self.X = X
        self.Y = Y
        self.M = M
        self.tag2idx = tag2idx
        self.idx2tag = {i: tag for tag, i in tag2idx.items()}
        if type(tokenizer) is str:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case='-cased' not in tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.for_mtl = for_mtl

    def __len__(self):
        return len(self.X)

    def _filter_whitespace(self, idx):
        """
        Ignore whitespace characters (e.g., newlines, spaces, tabs)

        :param idx:
        :return:
        """
        X = np.array(self.X[idx])
        Y = self.Y[idx]
        M = np.array(self.M[idx] if self.M else [1] * len(X))

        # transform to dense array
        Y = Y.toarray() if type(Y) is csr_matrix else np.array(Y)
        mask = np.array([i for i, w in enumerate(X) if w.strip()])

        if len(mask) > 0:
            return X[mask], Y[mask], M[mask]
        else:
            return X, Y, M

    def __getitem__(self, idx):

        X, Y, M = self._filter_whitespace(idx)

        # probabalistic labels
        is_proba = True if len(Y.shape) > 1 else False

        # add BERT special tokens
        X = np.concatenate((['[CLS]'], X, ['[SEP]']))
        M = np.concatenate(([0], M, [0]))
        if is_proba:
            pad = np.zeros((1, len(self.tag2idx)))
            Y = np.concatenate((pad, Y, pad))
        else:
            Y = np.concatenate((['X'], Y, ['X']))
            Y = np.array([self.tag2idx[t] for t in Y])

        x, y, m = [], [], []
        is_heads = []

        negation_heads = []

        # print(len(X))
        in_span = False
        last_label = None
        for i, (word, tag, mask) in enumerate(zip(X, Y, M)):
            tokens = self.tokenizer.tokenize(word) if word not in ('[CLS]', '[SEP]') else [word]
            tag = tag.reshape(1, -1) if is_proba else [tag]

            pad_len = len(tokens) - 1
            if pad_len != 0:
                pad = np.zeros((pad_len, Y.shape[-1])) if is_proba else [self.tag2idx['X']] * (len(tokens) - 1)
                tag = np.concatenate((tag, pad))

            is_head = [1] + [0] * (len(tokens) - 1)

            if mask:
                curr_label = np.argmax(tag)
                if not in_span or curr_label != last_label:
                    negation_heads.extend(is_head)
                    in_span = True
                else:
                    negation_heads.extend([0] * len(tokens))
            else:
                negation_heads.extend([0] * len(tokens))
                in_span = False
                last_label = None

            # if mask and not in_span:
            #     negation_heads.extend(is_head)
            #     in_span = True
            # elif mask and in_span:
            #     curr_label = np.argmax(tag)
            #     if curr_label != last_label:
            #         negation_heads.extend(is_head)
            #     else:
            #         negation_heads.extend([0] * len(tokens))
            # elif not mask:
            #     in_span = False
            #     negation_heads.extend([0] * len(tokens))
            # else:
            #     negation_heads.extend([0] * len(tokens))

            is_heads.extend(is_head)

            y.append(tag)
            m.extend([mask] * len(tag))
            x.extend(self.tokenizer.convert_tokens_to_ids(tokens))

            if mask:
                last_label = np.argmax(tag)

        y = np.concatenate(y)

        if len(x) > self.max_seq_len:
            x = np.concatenate((x[:self.max_seq_len - 1], x[-1:]))
            y = np.concatenate((y[:self.max_seq_len - 1], y[-1:]))
            m = np.concatenate((m[:self.max_seq_len - 1], m[-1:]))
            is_heads = np.concatenate((is_heads[:self.max_seq_len - 1], is_heads[-1:]))
            negation_heads = np.concatenate((negation_heads[:self.max_seq_len - 1], negation_heads[-1:]))
        # pdb.set_trace()
        # Y = np.array([self.idx2tag[t] for t in Y])
        if self.for_mtl:
            X_dict = {'X': X, 'x': x, 'is_heads': is_heads, 'mask': m, 'seq_lens': len(x),
                      'negation_heads': negation_heads}
            Y_dict = {'y': y}
            return X_dict, Y_dict
        else:
            return X, x, is_heads, m, y, len(x)


class ConllDataset(SequenceLabelingDataset):

    def __init__(self, fpath, tokenizer='bert-base-cased', max_seq_len=512):
        """
        Load CoNLL text dataset and tranform into BERT sequences

        :param fpath:
        :param tokenizer:
        :param max_seq_len:
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents = []
        tags = []
        for entry in entries:
            x = [line.split()[0] for line in entry.splitlines()]
            y = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(x)
            tags.append(y)

        tagset = ['X'] + sorted(set([t for seq in tags for t in seq]), key=lambda x: len(x))
        tag2idx = {tag: i for i, tag in enumerate(tagset)}

        SequenceLabelingDataset.__init__(self, sents, tags, tag2idx, tokenizer, max_seq_len)


class ProbaConllDataset(ProbaSequenceLabelingDataset):

    def __init__(self,
                 fpath: str,
                 tag2idx: Dict[str, int],
                 tokenizer: str = 'bert-base-cased',
                 max_seq_len: int = 512):
        """
        Load CoNLL-like text dataset and tranform into BERT sequences

        :param fpath:
        :param tokenizer:
        :param max_seq_len:
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
        X, Y, M = [], [], []

        # print('ENTRIES: ')
        for entry in entries:
            # print('entry: ', entry)
            words = [line.split()[0] for line in entry.splitlines()]
            y_proba = [[0.0] + list(map(float, line.split()[1:-1]))
                       for line in entry.splitlines()]
            # y_proba = [[0] if int(float(line.split()[1])) == 1 else [1] for line in entry.splitlines()]
            mask = [int(line.split()[-1]) for line in entry.splitlines()]
            X.append(words)
            Y.append(y_proba)
            M.append(mask)

        ProbaSequenceLabelingDataset.__init__(self, X, Y, tag2idx, tokenizer,
                                              max_seq_len, M, for_mtl=False)


class MTLProbaConllDataset(ProbaSequenceLabelingDataset):

    def __init__(self,
                 fpath: str,
                 tag2idx: Dict[str, int],
                 tokenizer: str = 'bert-base-cased',
                 max_seq_len: int = 512,
                 split='train',
                 task_name='',
                 name='dataset'):
        """
        Load CoNLL-like text dataset and tranform into BERT sequences

        :param fpath:
        :param tokenizer:
        :param max_seq_len:
        """
        self.split = split
        self.name = name
        self.task_name = task_name

        self.Y_dict = {task_name: None}  # hack to make MTL work
        entries = open(fpath, 'r').read().strip().split("\n\n")
        X, Y, M = [], [], []

        # print('ENTRIES: ')
        for entry in entries:
            # pdb.set_trace()
            words = []
            y_proba = []
            mask = []
            for line in entry.splitlines():

                line_split = line.split()
                if 'temporality' not in task_name and len(line_split) != 4:
                    continue
                elif 'temporality' in task_name and len(line_split) != 6:
                    continue
                words.append(line_split[0])

                # print([0.0] + list(map(float, line_split[1:-1])))
                y_proba.append([0.0] + list(map(float, line_split[1:-1])))
                # TODO(saelig) remove hack
                if sum(map(float, line_split[1:-1])) == 0:  # if no LF fired
                    mask.append(0)  # make sure this elem is masked
                else:
                    mask.append(int(line_split[-1]))

            if len(y_proba) == 0:
                continue
            X.append(words)
            Y.append(y_proba)
            M.append(mask)

        ProbaSequenceLabelingDataset.__init__(self, X, Y, tag2idx, tokenizer,
                                              max_seq_len, M, for_mtl=True)


def pad_seqs(batch):
    """

    :param batch:
    :return:
    """
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    X, Y = [], []
    for sample in batch:
        x = sample[1]
        y = sample[-2]
        # store data type so that y is the proper torch tensor type
        dtype = type(y[0])
        pad_len = maxlen - len(x)
        # pad to fixed max length
        x_pad = np.array([0] * pad_len)
        # y requires different padding if using probabalistic labels
        y_pad = np.zeros((pad_len, y.shape[1])) if len(y.shape) == 2 else np.zeros((pad_len,))
        X.append(np.concatenate((x, x_pad)))
        Y.append(np.concatenate((y, y_pad)))

        # X.append(torch.LongTensor(np.concatenate((x, x_pad))))
        # Y.append(torch.LongTensor(np.concatenate((y, y_pad))))

    # return words, X, is_heads, tags, Y, seqlens

    f = torch.LongTensor if dtype is np.int64 else torch.FloatTensor
    return words, f(X), is_heads, tags, f(Y), seqlens


def pad_seqs_with_mask(batch):
    """

    :param batch:
    :return:
    """
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    # tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    X, Y, M = [], [], []
    for sample in batch:
        _, x, _, m, y, _ = sample
        # store data type so that y is the proper torch tensor type
        dtype = type(y[0])
        pad_len = maxlen - len(x)
        # pad to fixed max length
        x_pad = np.array([0] * pad_len)
        # y requires different padding if using probabalistic labels
        y_pad = np.zeros((pad_len, y.shape[1])) if len(y.shape) == 2 else np.zeros((pad_len,))
        X.append(np.concatenate((x, x_pad)))
        Y.append(np.concatenate((y, y_pad)))
        M.append(np.concatenate((m, x_pad)))

        # X.append(torch.LongTensor(np.concatenate((x, x_pad))))
        # Y.append(torch.LongTensor(np.concatenate((y, y_pad))))

    # return words, X, is_heads, tags, Y, seqlens

    f = torch.LongTensor if dtype is np.int64 else torch.FloatTensor
    return words, f(X), is_heads, f(M), f(Y), seqlens


def pad_seqs_mtl(batch, task_name=''):
    X_batch = collections.defaultdict(list)
    Y_batch = collections.defaultdict(list)

    for x_dict, _ in batch:
        X_batch[f'{task_name}_seq_lens'].append(x_dict['seq_lens'])
        X_batch[f'{task_name}_X'].append(x_dict['X'])
    maxlen = np.array(X_batch[f'{task_name}_seq_lens']).max()

    for x_dict, y_dict in batch:
        x, mask = x_dict['x'], x_dict['mask']
        y = y_dict['y']
        dtype = type(y[0])

        pad_len = maxlen - len(x)
        x_pad = np.array([0] * pad_len)
        y_pad = np.zeros((pad_len, y.shape[1])) if len(y.shape) == 2 else np.zeros((pad_len,))

        X_batch[f'{task_name}_x'].append(np.concatenate((x, x_pad)))
        X_batch[f'{task_name}_mask'].append(np.concatenate((mask, x_pad)))
        X_batch[f'{task_name}_X'].append(x_dict['X'])
        X_batch[f'{task_name}_is_heads'].append(x_dict['is_heads'])
        X_batch[f'{task_name}_negation_heads'].append(np.concatenate((x_dict['negation_heads'], x_pad)))

        Y_batch[f'{task_name}'].append(np.concatenate((y, y_pad)))

    # pdb.set_trace()
    map_type = torch.LongTensor if dtype is np.int64 else torch.FloatTensor
    X_batch[f'{task_name}_x'] = torch.FloatTensor(X_batch[f'{task_name}_x'])
    X_batch[f'{task_name}_mask'] = map_type(X_batch[f'{task_name}_mask'])
    Y_batch[f'{task_name}'] = map_type(Y_batch[f'{task_name}'])

    X_batch[f'{task_name}_negation_heads'] = map_type(X_batch[f'{task_name}_negation_heads'])

    return dict(X_batch), dict(Y_batch)
