#
# Span matching
# TODO: Refactor this module
#
import re
from trove.dataloaders.contexts import Span, Sentence
from typing import List, Set, Dict, Tuple, Pattern, Match, Iterable, Callable

###############################################################################
#
# Span Matching
#
###############################################################################

def char_to_word_index(ci, sequence):
    """
    Given a character-level index (offset),
    return the index of the **word this char is in**
    """
    i = None
    for i, co in enumerate(sequence.char_offsets):
        if ci == co:
            return i
        elif ci < co:
            return i - 1
    return i


def get_word_index_span(char_offsets, sequence):
    char_start, char_end = char_offsets
    return (char_to_word_index(char_start, sequence),
            char_to_word_index(char_end, sequence))


def get_text(words, offsets):
    s = ''
    for i, term in zip(offsets, words):
        if len(s) == i:
            s += term
        elif len(s) < i:
            s += (' ' * (i - len(s))) + term
        else:
            raise Exception('text offset error')
    return s


def retokenize(words, offsets, split_on=r'''([/-])'''):
    """Apply secondary tokenization rule, e.g., split on hyphens
    and forward slashes.

    Parameters
    ----------
    words
    offsets
    split_on

    Returns
    -------

    """
    f_words, f_offsets = [], []
    for w, i in zip(words, offsets):

        tokens = [t for t in re.split(split_on, w) if t.strip()]
        if len(tokens) == 1:
            f_words.append(w)
            f_offsets.append(i)
        else:
            offset = i
            for j, t in enumerate(tokens):
                f_words.append(t)
                f_offsets.append(offset)
                offset += len(t)
    return f_words, f_offsets


def match_term(term, dictionary, case_sensitive, lemmatize=True):
    """

    Parameters
    ----------
    term
    dictionary
    case_sensitive
    lemmatize   Including lemmas improves performance slightly

    Returns
    -------

    """
    if (not case_sensitive and term.lower() in dictionary) or term in dictionary:
        return True
    if (case_sensitive and lemmatize) and term.rstrip('s').lower() in dictionary:
        return True
    elif (not case_sensitive and lemmatize) and term.rstrip('s') in dictionary:
        return True
    return False


def apply_matcher(words,
                  offsets,
                  dictionary,
                  max_ngrams=5,
                  longest_match_only=True,
                  case_sensitive = False,
                  split_on=None):
    """
    TODO: cleanup!
    """
    # covert to source char offsets
    text = get_text(words, offsets)

    matches = []
    for i in range(0, len(words)):
        match = None
        start = offsets[i]

        for j in range(i + 1, min(i + max_ngrams + 1, len(words) + 1)):
            end = offsets[j - 1] + len(words[j - 1])
            # term types: normalize whitespace & tokenized + whitespace
            for term in [
                re.sub(r'''\s{2,}''', ' ', text[start:end]).strip(),
                ' '.join([w for w in words[i:j] if w.strip()])
            ]:
                if match_term(term, dictionary, case_sensitive):
                    match = end
                    break

        if match:
            term = re.sub(r'''\s{2,}''', ' ', text[start:match]).strip()
            matches.append(([start, match], term))

    if longest_match_only:
        # sort on length then end char
        matches = sorted(matches, key=lambda x: x[0][-1], reverse=1)
        f_matches = []
        curr = None
        for m in matches:
            if curr is None:
                curr = m
                continue
            (i, j), _ = m
            if (i >= curr[0][0] and i <= curr[0][1]) and (j >= curr[0][0] and j <= curr[0][1]):
                pass
            else:
                f_matches.append(curr)
                curr = m
        if curr:
            f_matches.append(curr)
        return f_matches

    return matches


def overlaps(x, y):
    if x.start == x.stop or y.start == y.stop:
        return False
    return ((x.start < y.stop  and x.stop > y.start) or
            (x.stop  > y.start and y.stop > x.start))


def match_regex(rgx, span):
    """Return Span object for regex match"""
    m = re.search(rgx, span.text, re.I) if type(rgx) is str else rgx.search(span.text)
    if not m:
        return None
    i,j = m.span()
    if type(span) is Span:
        i += span.char_start
        j += span.char_start
        return Span(i, j-1, span.sentence)
    return Span(i, j-1, span)


def match_rgx(rgx: Pattern, sentence: Sentence) -> Dict[Tuple, Span]:
    """Match a regular expression to a sentence
    TODO: search over ngrams vs. entire sentence by default

    Parameters
    ----------
    rgx
    sentence

    Returns
    -------

    """
    matches = {}
    for match in rgx.finditer(sentence.text):
        start, end = match.span()
        span = Span(char_start=start, char_end=end - 1, sentence=sentence)
        matches[(start, end - 1, end - 1 - start)] = span
    return matches


def get_longest_matches(matches: Dict[Tuple, Span]) -> Iterable[Span]:

    mask = {}
    for key in sorted(matches.keys(), key=lambda x: x[-1], reverse=1):
        is_longest = True
        start, end, length = key
        span = matches[key]
        for j in range(start, end):
            if j not in mask:
                mask[j] = span
            else:
                is_longest = False
        if is_longest:
            yield span
