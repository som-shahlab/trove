"""
Labeling function implementation of  "A simple algorithm for identifying
abbreviation definitions in biomedical text"
Schwartz AS, Hearst MA
Computer Science Division, University of California,
Berkeley, Berkeley, CA 94720, USA. sariel@cs.berkeley.edu

Pac Symp Biocomput. 2003;:451-62.

http://psb.stanford.edu/psb-online/proceedings/psb03/abstracts/p451.html

TODO: Refactor

"""
import re
import collections
from trove.dataloaders.contexts import Span
from trove.labelers.labeling import (
    LabelingFunction,
    get_word_index_span,
    apply_matcher
)
from typing import List, Set, Dict

def is_short_form(text, min_length=2):
    """ Rule-based function for determining if a token is likely
    an abbreviation, acronym or other "short form" mention

    Parameters
    ----------
    text
    min_length

    Returns
    -------

    """
    accept_rgx = '[0-9A-Z-]{2,8}[s]*'
    reject_rgx = '([0-9]+/[0-9]+|[0-9]+[-][0-7]+)'

    keep = re.search(accept_rgx, text) is not None
    keep &= re.search(reject_rgx, text) is None
    keep &= not text.strip("-").isdigit()
    keep &= "," not in text
    keep &= len(text) < 15

    # reject if too short too short or contains lowercase single letters
    reject = (len(text) > 3 and not keep)
    reject |= (len(text) <= 3 and re.search("[/,+0-9-]", text) is not None)
    reject |= (len(text) < min_length)
    reject |= (len(text) <= min_length and text.islower())

    return False if reject else True


def get_parenthetical_short_forms(sentence):
    """Generator that returns indices of all words directly
    wrapped by parentheses or brackets.

    Parameters
    ----------
    sentence

    Returns
    -------

    """
    for i, _ in enumerate(sentence.words):
        if i > 0 and i < len(sentence.words) - 1:
            window = sentence.words[i - 1:i + 2]
            if window[0] == "(" and window[-1] == ")":
                if is_short_form(window[1]):
                    yield i


def extract_long_form(i, sentence, max_dup_chars=2):
    '''
    Search the left window for a candidate long-form sequence.
    Use the heuristic of "match first character" to guess long form
    '''
    short_form = sentence.words[i]
    left_window = [w for w in sentence.words[0:i]]

    # strip brackets/parentheses
    while left_window and left_window[-1] in ["(", "[", ":"]:
        left_window.pop()

    if len(left_window) == 0:
        return None

    # match longest seq to the left of our short form
    # that matches on starting character
    long_form = []
    char = short_form[0].lower()
    letters = [t[0].lower() for t in short_form]
    letters = [t for t in letters if t == char]
    letters = letters[0:min(len(letters), max_dup_chars)]

    matched = False

    for t in left_window[::-1]:
        if t[0] in "()[]-+,":
            break

        if len(letters) == 1 and t[0].lower() == letters[0]:
            long_form += [t]
            matched = True
            break

        elif len(letters) > 1 and t[0].lower() == letters[0]:
            long_form += [t]
            matched = True
            letters.pop(0)

        else:
            long_form += [t]

    # We didn't find the first letter of our short form, so
    # back-off and choose the longest contiguous noun phrase
    if (len(left_window) == len(long_form) and \
        letters[0] != t[0].lower() and \
        len(long_form[::-1]) > 1) or not matched:

        tags = list(zip(sentence.words[0:i - 1],
                        sentence.pos_tags[0:i - 1]))[::-1]
        noun_phrase = []

        while tags:
            t = tags.pop(0)
            if re.search("^(NN[PS]*|JJ)$", t[1]):
                noun_phrase.append(t)
            else:
                break

        if noun_phrase:
            long_form = list(zip(*noun_phrase))[0]

    # create candidate
    n = len(long_form[::-1])
    offsets = sentence.char_offsets[0:i - 1][-n:]
    char_start = min(offsets)
    words = sentence.words[0:i - 1][-n:]

    offsets = map(lambda x: len(x[0]) + x[1], zip(words, offsets))
    char_end = max(offsets)

    span = Span(char_start, char_end - 1, sentence)

    return Span(char_start, char_end - 1, sentence)


def get_short_form_index(cand_set):
    '''
    Build a short_form->long_form mapping for each document. Any
    short form (abbreviation, acronym, etc) that appears in parenthetical
    form is considered a "definition" and added to the index. These candidates
    are then used to augment the features of future mentions with the same
    surface form.
    '''

    sf_index = {}
    for doc in cand_set:

        for sent in doc.sentences:
            for i in get_parenthetical_short_forms(sent):
                short_form = sent.words[i]
                long_form_cand = extract_long_form(i, sent)

                if not long_form_cand:
                    continue
                if doc.doc_id not in sf_index:
                    sf_index[doc.doc_id] = {}
                if short_form not in sf_index[doc.doc_id]:
                    sf_index[doc.doc_id][short_form] = []
                sf_index[doc.doc_id][short_form] += [long_form_cand]

    return sf_index


class SchwartzHearstLabelingFunction(LabelingFunction):
    """
    Schwartz-Hearst algorithm (2003)

    This builds a per-document dictionary of short-form<->long-forms that
    match the semantic type defined by the provided ontology/dictionary.

    """
    def __init__(self,
                 name:str,
                 dictionary: Set[str],
                 label:int,
                 stopwords: Set[str] = None):

        super().__init__(name, label)
        self._index = {}
        self.dictionary = dictionary
        self.stopwords = set() if not stopwords else stopwords

    def _doc_term_forms(self, doc):
        """

        :param doc:
        :return:
        """
        # return cached dictionary
        if doc.name in self._index:
            return self._index[doc.name]

        abbrv_map = collections.defaultdict(set)
        for sent in doc.sentences:
            for i in get_parenthetical_short_forms(sent):
                short_form = sent.words[i]
                long_form = extract_long_form(i, sent)

                if not long_form:
                    continue

                abbrv_map[short_form].add(long_form)

        # map each short form to a class label
        term_labels = {}
        for sf in abbrv_map:
            label = None
            for term in abbrv_map[sf]:
                if term.text in self.dictionary or term.text.lower() in self.dictionary:
                    label = self.label
                    break

            # if any long form is in our class dictionaries,
            # treat this as a synset for the class label
            if label:
                term_labels[sf] = label
                for term in abbrv_map[sf]:
                    term_labels[term.text.lower()] = label

        # cache
        self._index[doc.name] = term_labels
        return self._index[doc.name]

    def __call__(self, sentence):

        # extract abbreviation definitions or pull from cache
        doc_term_dict = self._doc_term_forms(sentence.document)

        # get all string matches
        m = apply_matcher(sentence.words,
                          sentence.char_offsets,
                          doc_term_dict,
                          max_ngrams=5,
                          split_on=None,
                          longest_match_only=False)

        # no matches
        if not m:
            return {}

        # for each span, assign label
        L = {}
        for (char_start, char_end), term in m:
            if term in self.stopwords:
                continue
            start, end = (
                get_word_index_span((char_start, char_end - 1), sentence)
            )
            for i in range(start, end + 1):
                L[i] = self.label
        return L
