"""
spaCy Clinical Text Tokenizer & Sentence Boundary Detection (SBD)

Tested on clinical notes from:
- MIMIC-III
- Stanford Health Care
- THYME
- i2b2/n2c2 2009 Medications

"""
import re
import os
import toolz
import spacy
import logging
from functools import partial
from collections import defaultdict
from spacy.tokenizer import Tokenizer
from spacy.pipeline import SentenceSegmenter
from spacy.symbols import ORTH, LEMMA, POS, TAG
from typing import List, Set, Dict, Tuple, Optional, Union, Callable, Generator


logger = logging.getLogger(__name__)

###############################################################################
#
# Sentence Boundary Detection
#
###############################################################################

def split_on_punct(doc):
    """
    Default punctuation-based SBD

    :param doc:
    :return:
    """
    start = 0
    seen_period = False
    for i, word in enumerate(doc):
        if seen_period and not word.is_punct:
            yield doc[start:word.i]
            start = word.i
            seen_period = False

        elif word.text in ['.', '!', '?']:
            seen_period = True
    if start < len(doc):
        yield doc[start:len(doc)]


def split_on_rgx(sentences, doc, rgx, threshold=250, sent_match=None):
    """
    Match split tokens using provided regex.

    :param sentences:
    :param doc:
    :param rgx:
    :param threshold:
    :return:
    """
    splits = []
    for sent in sentences:
        if len(sent.text) >= threshold and not sent_match or \
                (len(sent.text) >= threshold and sent_match(sent)):
            idxs = [sent[0].i] + [word.i for word in sent \
                                  if rgx.search(word.text)] + [sent[-1].i + 1]
            idxs = sorted(list(set(idxs)))
            for i in range(len(idxs) - 1):
                splits.append(doc[idxs[i]:idxs[i + 1]])
        else:
            splits.append(sent)
    return splits



def split_on_phrase_rgx(sentences, doc, rgx, threshold=250):
    """
    Split sentence on phrase regex

    :param sentences:
    :param doc:
    :param rgx:
    :param threshold:
    :return:
    """
    splits = []
    for sent in sentences:

        matches = re.findall(rgx, sent.text)
        if len(sent.text) >= threshold and matches:
            offset = sent[0].idx
            # split up sentence
            m_idxs = set()
            for m in matches:
                m_idxs.add(sent.text.index(m) + offset)

            idxs = [sent[0].i]
            idxs += [word.i for word in sent if word.idx in m_idxs]
            idxs += [sent[-1].i + 1]

            idxs = sorted(list(set(idxs)))
            for i in range(len(idxs) - 1):
                splits.append(doc[idxs[i]:idxs[i + 1]])
        else:
            splits.append(sent)
    return splits


def merge_sentences(doc, sents, merge_terms):
    """
    Use a collection of bigrams (either from a dictionary of bigrams,
    corpus association weights, etc.) to define word pairs that cannot
    be split across sentences.

    TODO: Clean this up!

    :param doc:
    :param idxs:
    :param merge_terms:
    :return:
    """
    # terms that can never end a sentence
    non_terminals = {
        ',', '-', '(', '=', '/', 'mrs.', 'mr.', 'ms.',
        'dr.', 'at', 'with', 'and', 'the', 'is', 's/p'
    }

    # word indices
    sequences = [[
        word.i for word in sent if word.text.strip()] for sent in sents
    ]
    sequences = [idxs for idxs in sequences if len(idxs) != 0]

    stack = [sequences.pop(0)]
    for seq in sequences:
        i = stack[-1][-1]
        j = seq[0]
        text = re.sub(r'''\s{2,}|\n''', ' ', doc[i:j + 1].text).lower().strip()
        if text in merge_terms or doc[i].text in non_terminals:
            stack[-1].extend(seq)
        else:
            stack.append(seq)

    # replace missing indices (whitespace)
    end = 0
    sentences = []
    stack[0] = [0] + stack[0]
    for i in range(len(stack)-1):
        s = list(sorted(set(stack[i] + [stack[i+1][0]])))
        sentences.append(doc[min(s):max(s)])
        end = max(s)

    if doc[end:]:
       sentences.append(doc[end:])

    return sentences


def ct_sbd_rules(doc, merge_terms=None, max_sent_len=None):
    """
    Split sentences if they don't meet certain char length thresholds.
    This splits on 3 and 2 character whitespace tokens and bulleted lists.

    :param doc:
    :param threshold:
    :return:
    """
    merge_terms = {} if not merge_terms else merge_terms

    sents = [sent for sent in split_on_punct(doc)]
    sents = split_on_rgx(sents, doc, re.compile("\s{2,}"), threshold=250)
    sents = split_on_rgx(sents, doc, re.compile("\s{1,}"), threshold=100,
                         sent_match=lambda x: x.text.count(":") > 2)
    sents = split_on_rgx(sents, doc, re.compile("[•](?![CF])"), threshold=10)

    # combine sentences based on a list terms that cannot split
    sents = merge_sentences(doc, sents, merge_terms)

    # header matches
    # TODO -- this has an off-by-one error
    #rgx = r'''\s{2,}((?:(?:[A-Z][A-Za-z]+\s){1,4}(?:[A-Za-z]+))[:])'''
    #sents = split_on_phrase_rgx(sents, doc, re.compile(rgx), threshold=1)

    # force sentences to have a max length
    if max_sent_len:
        splits = []
        for s in sents:
            idxs = [word.i for word in s]
            if len(idxs) > max_sent_len:
                parts = list(toolz.partition_all(max_sent_len, idxs))
                for p in parts:
                    seq = doc[p[0]:p[-1] + 1]
                    splits.append(seq)
            else:
                seq = doc[idxs[0]:idxs[-1] + 1]
                splits.append(seq)

        sents = splits

    for s in sents:
        yield s


###############################################################################
#
# Tokenization
#
###############################################################################

def load_special_cases(filelist):
    """
    Load manually defined special cases (including any lexical metadata
    like POS tag)

    :param filelist:
    :return:
    """
    spacy_symbols = {'ORTH': ORTH, 'TAG': TAG, 'LEMMA': LEMMA}

    for fpath in filelist:
        for i, row in enumerate(open(fpath, "rU").read().splitlines()):
            row = row.strip().split("\t")
            if not row:
                continue
            if i == 0:
                header = row
                continue
            row = dict(zip(header, row))
            attribs = {spacy_symbols[key]: value
                       for key, value in row.items() if key in spacy_symbols}
            attribs[ORTH] = row['TERM'] if ORTH not in attribs \
                else attribs[ORTH]
            yield row['TERM'], [attribs]


def add_special_cases(tokenizer, special_cases):
    """
    Load special cases for the tokenizer.
    These are largely clinical abbreviations/acronyms.

    :param tokenizer:
    :param special_cases:
    :return:
    """
    for term, attrib in load_special_cases(special_cases):
        tokenizer.add_special_case(term, attrib)


def build_token_match_rgx():
    """
    Build accept & reject patterns for individual tokens. Preserving certain
    token groupings (e.g., lab values) is useful for preventing false positives
    during sentence boundary detection.

    :return:
    """
    # accept tokenization
    include_rgx = [
        r'''^[(][0-9]''',            # ignore some head/tail punctuation cases with leading paranthesis
        r'''[/][0-9]+[,]$''',
        r'''[0-9]+[/][0-9]+[.,]$''',  # fix dates with trailing punctuation: 01/01/2001,
    ]

    # override tokenization
    exclude_rgx = [
        r'''^[0-9]{1,3}[.][0-9]{1,2}[/][0-9]{1,3}[.][0-9]{1,2}$''', # ratio of floats: 0.3/0.7
        r'''^[-]*[0-9]{1,3}[.][0-9]{1,4}$''',       # 100.02 -1.002
        r'''^([0-9]{3}[.]){2}[0-9]{4}$''',          # Phone numbers, 555.555.5555
        r'''^[A-Z]*[0-9]+[.][0-9A-Z]+$''',          # ICD9 codes: 136.9BJ
        r'''^[0-9]+[.][0-9]+([%]|mm|cm|mg|ml)$''',  # measurement 1.0mm
        r'''[0-9]+[.][0-9]+[-][0-9]+[.][0-9]+''',   # ignore range/intervals 0.1-0.4 mg
        r'''^[0-9]+[.][0-9]+$''',
        r'''^([A-Z][\.]|[1-9][0-9]*[\.)])$'''       # list item or single letter (often middle initial)
        r'''[0-9]+[/][0-9]+''',                     # fractions, blood pressure readings, etc.: 1/2 120/80
        #r'''^[A-Za-z][/][A-Za-z]([/][A-Za-z])*$''', # skip abbreviations of the form: n/v, c/d/i

        # date time expressions
        r'''([01][0-9]/[0-3][0-9])''',              # 11/12
        r'''[0-1]{0,1}[0-9][/]([3][01]|[12][0-9]|[0-9])[/]((19|20)[0-9]{2}|[0-3][0-9])\b''',    # 1/11/2000
        r'''http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+''', # URL
        r'''^([0-9]{1,2}|[A-Z])[).]$''',                                                        # List items: 1. 1) A.
        r'''[0-2][0-9][:][0-9]{2}[:][0-9]{2}[.][0-9]''',                                        # times 11:09:00.0

        # lab values
        r'''[A-Za-z()]+[-<]{1,2}[0-9]{1,2}[.][0-9]{1,2}[*#]{0,2}''',  # cTropnT-<0.01 |  HCT-26.7*  | INR(PT)-1.3
        r'''([0-9]+[-][0-9]+[-][0-9]+)|([0-9]+[-][0-9]+)'''           # dates
    ]

    return re.compile('|'.join(include_rgx)), \
           re.compile('|'.join(exclude_rgx))


include_rgx, exclude_rgx = build_token_match_rgx()

def token_match(s):
    if include_rgx.search(s):
        return False
    elif exclude_rgx.search(s):
        return True
    return False


def ct_tokenizer(nlp):
    """
    Clinical Note Tokenizer

    - Keep prefix/suffix/infix regexes as simple as possible
    - Move token complexity to special cases when possible
    - token_match exceptions are can be *order dependant* so use caution

    :param nlp:
    :return:
    """
    prefix_re = re.compile(r'''^([\["'()*+-?/<>#%]+|[><][=])+''')
    suffix_re = re.compile(r'''([\]"'),-.:;*]|'s)$''')
    infix_re  = re.compile(r'''[%(),-./;=?]+''')  # spaCy SBD break w/o [.]

    tokenizer = Tokenizer(nlp.vocab,
                          prefix_search=prefix_re.search,
                          suffix_search=suffix_re.search,
                          infix_finditer=infix_re.finditer,
                          token_match=token_match)

    special_cases = [
        fp.format(os.path.dirname(__file__))
        for fp in ["{}/specialist_special_cases.txt", "{}/special_cases.txt"]
    ]
    add_special_cases(tokenizer, special_cases)
    return tokenizer


###################################################################
#
# Parsers
#
###################################################################

def parse_doc(doc: spacy.tokens.Doc,
              disable: Set[str] = None,
              keep_whitespace: bool = False):
    """
    Given a parsed document, generate a parsed Sentence object

    :param doc:
    :param disable:
    :return:
    """
    disable = {"ner", "parser", "tagger", "lemmatizer"} if not disable \
        else disable
    for position, sent in enumerate(doc.sents):
        parts = defaultdict(list)

        for i, token in enumerate(sent):

            # TODO: figure out a better way to deal with whitespace
            text = str(sent.text)
            parts['newlines'] = [m.span()[0] for m in re.finditer(r'''(\n)''', text)]

            if not keep_whitespace and not token.text.strip():
                continue

            parts['words'].append(token.text)
            parts['abs_char_offsets'].append(token.idx)

            # optional NLP tags
            if "lemmatizer" not in disable:
                parts['lemmas'].append(token.lemma_)
            if "tagger" not in disable:
                parts['pos_tags'].append(token.tag_)
            if "ner" not in disable:
                parts['ner_tags'].append(
                    token.ent_type_ if token.ent_type_ else 'O'
                )
            if "parser" not in disable:
                head_idx = 0 if token.head is token else \
                    token.head.i - sent[0].i + 1
                parts['dep_parents'].append(head_idx)
                parts['dep_labels'].append(token.dep_)

        # sentence is all whitespace
        if not parts['words']:
            continue

        parts['i'] = position
        yield parts


def get_parser(disable: List[str] = None ,
               lang: str = 'en',
               merge_terms: Optional[Set] = None,
               max_sent_len: Optional[int] = None) -> Callable:
    """spaCy clinical text parser

    Parameters
    ----------
    disable
    lang
    merge_terms
    max_sent_len

    Returns
    -------

    """
    disable = ["ner", "parser", "tagger", "lemmatizer"] if not disable \
        else disable
    merge_terms = {} if not merge_terms else merge_terms

    nlp = spacy.load(lang, disable=disable)
    nlp.tokenizer = ct_tokenizer(nlp)

    sbd_func = partial(ct_sbd_rules,
                       merge_terms=merge_terms,
                       max_sent_len=max_sent_len)

    sbd = SentenceSegmenter(nlp.vocab, strategy=sbd_func)
    nlp.add_pipe(sbd)
    return nlp
