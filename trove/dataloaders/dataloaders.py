import re
import os
import gzip
import json
import glob
import numpy as np
import logging
import itertools
import collections
from trove.dataloaders.contexts import Document, Sentence, Annotation

logger = logging.getLogger(__name__)


def parse_doc(d) -> Document:
    """
    Convert JSON into container objects. Transforming to
    Document/Sentence objects comes at ~13% overhead.
    """
    sents = [Sentence(**s) for s in d['sentences']]
    doc = Document(d['name'], sents)
    if 'metadata' in d:
        for key,value in d['metadata'].items():
            doc.props[key] = value
    return doc


class DocumentLoader:

    def __init__(self, fpath):
        self.fpath = fpath
        self.formatter = parse_doc

    def filelist(self):
        return glob.glob(f'{self.fpath}/*.json') \
            if os.path.isdir(self.fpath) else [self.fpath]

    def __iter__(self):
        for fpath in self.filelist():
            fopen = gzip.open if fpath.split(".")[-1] == 'gz' else open
            with fopen(fpath, 'rb') as fp:
                for line in fp:
                    yield self.formatter(json.loads(line))


def load_json_dataset(fpath,
                      tokenizer,
                      tag_fmt = 'IO',
                      contiguous_only = False):
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


def retokenize(sent, tokenizer, subword='##'):
    """
    Given a default tokenization, compute absolute character offsets for
    a new tokenization (e.g., BPE). By convention, wordpiece tokens are
    prefixed by ##.

    """
    tokens = []
    abs_char_offsets = []

    for i in range(len(sent.words)):
        toks = tokenizer.tokenize(sent.words[i])
        offsets = [sent.abs_char_offsets[i]]
        for w in toks[0:-1]:
            offsets.append(
                len(w if w[:len(subword)] != subword else w[len(subword):]) + offsets[-1]
            )
        abs_char_offsets.extend(offsets)
        tokens.extend(toks)

    return tokens, abs_char_offsets


def tokens_to_tags(sent,
                   entities,
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

    toks, abs_char_offsets = retokenize(sent, tokenizer) if tokenizer \
        else (sent.words, sent.abs_char_offsets)

    # truncate long sequences
    if len(toks) > max_seq_len - 2:
        toks = toks[0:max_seq_len - 2]
        abs_char_offsets = abs_char_offsets[0:max_seq_len - 2]

    # use original tokenization to assign token heads
    is_heads = [1 if i in sent.abs_char_offsets else 0 for i in abs_char_offsets]
    tags = ['O'] * len(toks)

    errs = 0
    for entity in entities:

        # currently we only support contiguous entity spans
        if len(entity.span) != 1:
            logger.warning(f"Non-contiguous entities not supported {entity} {sent.document.name}")
            continue

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
                logger.warning(f"Tokenization Error: Token is not a head token {entity} {sent.document.name}")
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
        Convert Document objects with a corresponding
        entity set into tagged sequences

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