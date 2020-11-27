import re
import itertools
import numpy as np
from trove.labelers.matchers import (
    apply_matcher,
    get_word_index_span,
    match_rgx,
    get_longest_matches
)
from trove.dataloaders.contexts import Span, Sentence
from typing import List, Set, Dict, Tuple, Pattern, Match, Iterable, Callable


###############################################################################
#
# Dictionary/Ontology Labeling Functions
#
###############################################################################

class LabelingFunction:

    def __init__(self, name, label):
        self.name = name
        self.label = label


class DictionaryLabelingFunction(LabelingFunction):
    """
    Labeling function that applies a dictionary to a sentence and outputs
    span labels. These spans are mapped to tokenized words to generate labels.
    """
    def __init__(self,
                 name:str,
                 dictionary: Set[str],
                 label:int,
                 case_sensitive: bool = False,
                 max_ngrams:int = 4,
                 stopwords = None):
        """

        Parameters
        ----------
        name
        dictionary
        label
        case_sensitive
        max_ngrams
        stopwords
        """
        super().__init__(name, label)
        self.case_sensitive = case_sensitive
        self.dictionary = dictionary
        self.max_ngrams = max_ngrams
        self.stopwords = {} if not stopwords else stopwords

    def __call__(self, sentence):
        """

        Parameters
        ----------
        sentence

        Returns
        -------

        """
        m = apply_matcher(sentence.words,
                          sentence.char_offsets,
                          self.dictionary,
                          max_ngrams=self.max_ngrams,
                          longest_match_only=False,
                          case_sensitive=self.case_sensitive)

        L = {}
        for (char_start, char_end), term in m:
            if term.lower() in self.stopwords or term in self.stopwords:
                continue
            start, end = get_word_index_span(
                (char_start, char_end - 1), sentence
            )
            for i in range(start, end + 1):
                L[i] = self.label
        return L


class OntologyLabelingFunction(LabelingFunction):
    """
    This labeling function assumes strings can match to multiple possible
    labels as defined by some ontology (a collection of typed dictionaries).
    An ontology is encoded as a dictionary mapping to class probabilities.

     term -> [0.5, 0.5]

    For emitting labels, we use the following rules by default:
     - Find the longest dictionary match (up to max_ngrams)
     - Return the most probable class
     - If all classes are equiprobable, abstain

    TODO:
    - Have LFs directly emit probabilities

    """
    def __init__(self,
                 name: str,
                 ontology: Set[str],
                 case_sensitive: bool = False,
                 max_ngrams: int = 4,
                 stopwords = None) -> None:

        super().__init__(name, None)
        self.max_ngrams = max_ngrams
        self.case_sensitive = case_sensitive
        self.stopwords = {} if not stopwords else stopwords

        # cache labels
        self._labels = {}
        for term, proba in ontology.items():
            # using numpy dtype causes a 60% slowdown with joblib
            self._labels[term] = None if np.all(proba == 1.0 / len(proba)) \
                else int(np.argmax(proba) + 1)
        self.ontology = frozenset(ontology)

    def _get_term_label(self, t):

        for key in [t, t.lower(), t.rstrip('s'), t + 's']:
            if key in self.stopwords:
                return self.stopwords[key]
            if key in self._labels:
                return self._labels[key]
        return None

    def __call__(self, sentence:Sentence) -> Dict[int, int]:

        matches = apply_matcher(sentence.words,
                                sentence.char_offsets,
                                self.ontology,
                                max_ngrams=self.max_ngrams,
                                longest_match_only=True,
                                case_sensitive=self.case_sensitive)
        matches = sorted(matches, key=lambda x:x[0], reverse=0)

        L = {}
        for (char_start, char_end), term in matches:
            label = self._get_term_label(term)

            # None labels are treated as abstains
            if not label:
                continue

            start, end = get_word_index_span(
                (char_start, char_end - 1), sentence
            )
            for i in range(start, end + 1):
                L[i] = label
        return L


class SlotFilledOntologyLabelingFunction(LabelingFunction):
    """
    Same as Ontology labeling function, except support slot-filled matches.
    This is useful for mentions that occur in simple patterns, e.g.,
        Tylenol (Acetaminophen)

    TODO: Refactor

    """
    def __init__(self,
                 name: str,
                 ontology: Set[str],
                 slot_patterns: List[str] = None,
                 case_sensitive: bool = False,
                 max_ngrams: int = 4,
                 stopwords=None,
                 span_rule=None) -> None:
        """

        Parameters
        ----------
        name
        ontology
        slot_patterns
        case_sensitive
        max_ngrams
        stopwords
        span_rule
        """
        super().__init__(name, None)
        self.max_ngrams = max_ngrams
        self.case_sensitive = case_sensitive
        self.stopwords = {} if not stopwords else stopwords
        self.span_rule = span_rule

        # cache labels
        self._labels = {}
        for term in ontology:
            proba = ontology[term]
            self._labels[term] = None if np.all(proba == 1.0 / len(proba)) \
                else int(np.argmax(proba) + 1)

        self.ontology = frozenset(ontology)
        self.slot_rgxs = slot_patterns if slot_patterns else []

    def _get_term_label(self, t):
        """

        Parameters
        ----------
        t

        Returns
        -------

        """
        for key in [t, t.lower(), t.rstrip('s'), t + 's']:
            if key in self.stopwords:
                return self.stopwords[key]
            if key in self._labels:
                return self._labels[key]
        return None

    def _merge_matches(self, matches):
        """ Merge all contiguous spans with the same label.

        Parameters
        ----------
        matches

        Returns
        -------

        """

        terms = [m[-1] for m in matches]
        labels = [self._get_term_label(m[-1]) for m in matches]

        matches = list(zip(matches, labels))
        merged = []

        i = 0
        while len(matches) - 1 > 0:
            (s1, t1), y1 = matches[i]
            (s2, t2), y2 = matches[i+1]
            # merge
            if s2[0] - s1[-1] == 1 and y1 == y2:
                m = [([s1[0], s2[-1]], f'{t1} {t2}'), y1]
                matches = [m] + matches[i+2:]
            else:
                merged.append(matches.pop(0))

        if matches:
            merged.append(matches.pop(0))

        merged, labels = zip(*merged)

        terms_b = [m[-1] for m in merged]
        assert " ".join(terms) == " ".join(terms_b)

        return merged, labels

    def __call__(self, sentence: Sentence) -> Dict[int, int]:
        """

        Parameters
        ----------
        sentence

        Returns
        -------

        """
        matches = apply_matcher(sentence.words,
                                sentence.char_offsets,
                                self.ontology,
                                max_ngrams=self.max_ngrams,
                                longest_match_only=True,
                                case_sensitive=self.case_sensitive)

        matches = sorted(matches, key=lambda x: x[0], reverse=0)
        if not matches:
            return {}

        matches, labels = self._merge_matches(matches)
        terms = [m[-1] for m in matches]

        # Slot-filled matches
        f_matches = []
        mask = np.array([0] * len(matches))
        for slot in self.slot_rgxs:
            n_args = slot.count('{}')
            args = list(zip(terms,labels))

            for i in range(len(args) - n_args + 1):

                # skip arguments that are already matched
                if 1 in mask[i:i+n_args]:
                    continue

                xs,ys = zip(*args[i:i+n_args])

                # HACK - positive classes only
                if None in ys or 2 in ys:
                    continue

                rgx = re.compile(slot.format(*xs), re.I)
                m = match_rgx(rgx, sentence)
                if m:
                    m = list(m.items())[0]
                    span = list(m[0][0:2])
                    span[-1] += 1
                    m = tuple([span, m[-1].text])
                    # expand the argument matches to this span
                    mask[i:i + n_args] = 1
                    f_matches.append((m, np.unique(ys)[0]))

        # add slot filled matches
        matches = [m for i, m in zip(mask, matches) if i == 0]
        labels = [y for i, y in zip(mask, labels) if i == 0]
        for m,y in f_matches:
            matches.append(m)
            labels.append(y)

        flip = False
        L = {}
        for ((char_start, char_end), term), label in zip(matches, labels):
            key = term.lower() if term.lower() in self._labels else term

            # None labels are treated as abstains
            if not label:
                continue

            # check span-specific rules
            if self.span_rule and label == 1:
                span = Span(char_start, char_end - 1, sentence)
                if self.span_rule(span):
                    label = 2
                    flip = True

            if term.lower() in self.stopwords or term in self.stopwords:
                label = 2
                #label = self.stopwords[key]

            start, end = get_word_index_span(
                (char_start, char_end - 1), sentence
            )
            for i in range(start, end + 1):
                L[i] = label

            flip = False

        return L

###############################################################################
#
# Regular Expression Labeling Functions
#
###############################################################################

class RegexEachLabelingFunction(LabelingFunction):
    """
    Use regular expressions to label tokens in a sentence.
    """
    def __init__(self, name, regexes, label):
        super().__init__(name, label)
        self.regexes = [
            re.compile(rgx) if type(rgx) is str else rgx for rgx in regexes
        ]

    def __call__(self, sentence):
        L = {}
        for i,word in enumerate(sentence.words):
            for rgx in self.regexes:
                if re.search(rgx, word):
                    L[i] = self.label
                    break
        return L


class RegexLabelingFunction(LabelingFunction):
    """
    Use regular expressions to label tokens in a sentence.
    """
    def __init__(self, name, regexes, label):
        super().__init__(name, label)
        self.regexes = [
            re.compile(rgx) if type(rgx) is str else rgx for rgx in regexes
        ]

    def __call__(self, sentence):
        matches = {}
        for rgx in self.regexes:
            matches.update(match_rgx(rgx, sentence))
        matches = list(get_longest_matches(matches))

        return {i:self.label for span in matches \
                for i in range(span.get_word_start(), span.get_word_end() + 1)}



###############################################################################
#
# SynSet Labeling Functions
#
###############################################################################

class SynSetLabelingFunction(LabelingFunction):
    """
    Given a map of TERM -> {t \in SYNONYMS}, if the TERM AND any t
    appear in document, label as a positive instance of the entity.
    """
    def __init__(self,
                 name: str,
                 synset: Dict[str, Set[str]],
                 label: int,
                 stopwords: Set[str] = None) -> None:

        super().__init__(name, label)
        self.synset = synset
        self.stopwords = set() if not stopwords else stopwords

    def __call__(self,
                 sentence: Sentence) -> Dict[int, int]:

        text = sentence.document.text
        v = {}
        for i, w in enumerate(sentence.words):
            if len(w) == 1:
                continue
            if w in self.synset:
                # check if any synset term appears in the document text
                for term in self.synset[w]:
                    if term in self.stopwords:
                        continue
                    # if so, label this word
                    if term in text or term.lower() in text.lower():
                        v[i] = self.label
                        break
        return v


###############################################################################
#
# Misc Labeling Functions
#
###############################################################################

class WordGraphLabelingFunction(LabelingFunction):

    def __init__(self, name, graph, label, min_length=3, sw = None):
        self.G = graph
        self.name = name
        self.label = label
        self.sw = sw if sw else {}
        self.min_length = min_length

    def _get_contiguous_spans(self, data):
        splits = []
        for i in range(len(data)-1):
            if data[i+1] - data[i] == 1:
                continue
            splits.append(i+1)
        splits = [0] + splits + [len(data)]
        return [data[splits[i]:splits[i+1]] for i in range(len(splits)-1)]

    def __call__(self, sentence):
        # search over bigram edges to assemble span
        tokens = sentence.words

        L = [0] * len(tokens)
        for i in range(len(tokens)-1):
            if tokens[i] in self.G or tokens[i].lower() in self.G:
                head = tokens[i] if tokens[i] in self.G else tokens[i].lower()
                if head in self.sw:
                    continue
                if tokens[i+1] in self.G[head] or tokens[i+1].lower() in self.G[head]:
                    tail = tokens[i+1] if tokens[i+1] in self.G[head] else tokens[i+1].lower()
                    if tail in self.sw:
                        continue
                    L[i] = self.label
                    L[i+1] = self.label

        L = {i:y for i,y in enumerate(L) if y != 0}
        spans = list(L.keys())

        spans = self._get_contiguous_spans(spans)
        spans = list(itertools.chain.from_iterable([s for s in spans if len(s) >= self.min_length]))
        return {i:L[i] for i in spans}