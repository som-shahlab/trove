import re
import csv
from collections import defaultdict
import numpy as np
from .helpers import *
from .taggers import Tagger
from scipy.stats import mode


class NegEx(object):
    '''
    NegEx

    Chapman, Wendy W., et al. "A simple algorithm for identifying negated
    findings and diseases in discharge summaries." Journal of biomedical
    informatics 34.5 (2001): 301-310.

    '''
    def __init__(self,data_root='supervision/dicts/negex'):
        self.data_root = data_root
        self.filename = "negex_multilingual_lexicon-en-de-fr-sv.csv"
        self.dictionary = NegEx.load("{}/{}".format(self.data_root,
                                                    self.filename))
        self.rgxs = NegEx.build_regexs(self.dictionary)

    def negation(self, span, category, direction, window=3):
        """
        Return any matched negex phrases

        :param span:
        :param category:
        :param direction:
        :param window:
        :return:
        """
        rgx = self.rgxs[category][direction]
        if not rgx.pattern:
            return None

        cxt = get_left_span(span, window=window) if direction == 'left' \
            else get_right_span(span, window=window)

        m = rgx.findall(cxt.text)
        return m if m else None

    def is_negated(self, span, category, direction, window=3):
        """
        Boolean test for negated spans

        :param span:
        :param category:
        :param direction:
        :param window:
        :return:
        """
        rgx = self.rgxs[category][direction]
        if not rgx.pattern:
            return False

        negation_match = self.negation(span, category, direction, window)
        return True if negation_match else False

    def all_negations(self, span, window=3):

        ntypes = []
        for category in self.rgxs:
            for direction in self.rgxs[category]:
                m = self.negation(span, category, direction, window)
                if m:
                    ntypes.append((category, direction, m))

        return ntypes

    @staticmethod
    def build_regexs(dictionary):
        """

        :param dictionary:
        :return:
        """
        rgxs = defaultdict(dict)
        for category in dictionary:
            fwd = [t["term"] for t in dictionary[category] \
                   if t['direction'] in ['forward', 'bidirectional']]
            bwd = [t["term"] for t in dictionary[category] \
                   if t['direction'] in ['backward', 'bidirectional']]
            rgxs[category]['left'] = "|".join(sorted(fwd, key=len, reverse=1))
            rgxs[category]['right'] = "|".join(sorted(bwd, key=len, reverse=1))

            if not rgxs[category]['left']:
                del rgxs[category]['left']
            if not rgxs[category]['right']:
                del rgxs[category]['right']
            for direction in rgxs[category]:
                p = rgxs[category][direction]
                rgxs[category][direction] = \
                    re.compile(r"({})(\b|$)".format(p), flags=re.I)

        return rgxs

    @staticmethod
    def load(filename):
        '''
        Load negex definitions
        :param filename:
        :return:
        '''
        negex = defaultdict(list)
        with open(filename, 'rU') as of:
            reader = csv.reader(of, delimiter=',')
            for row in reader:
                term = row[0]
                category = row[30]
                direction = row[32]
                if category == 'definiteNegatedExistence':
                    negex['definite'].append({'term': term,
                                              'direction': direction})
                elif category == 'probableNegatedExistence':
                    negex['probable'].append({'term': term,
                                              'direction': direction})
                elif category == 'pseudoNegation':
                    negex['pseudo'].append({'term': term,
                                            'direction': direction})
        return negex

###############################################################################
#
# Negation Tagger
#
###############################################################################

class NegExTagger(Tagger):

    def __init__(self, targets, data_root, label_reduction='or'):
        """
        label_reduction:  or|mv
        """
        self.targets = targets
        self.negex = NegEx(data_root=data_root)
        self.label_reduction = label_reduction

        # map negative types to output labels (1:True, 2:False)
        self.class_map = {
            'definite': 1,
            'probable': 1,
            'pseudo': 2
        }

        # LF names
        self.header = []
        for name in sorted(self.negex.rgxs):
            for cxt in sorted(self.negex.rgxs[name]):
                self.header.append(f'LF_{name}_{cxt}')

    def _apply_lfs(self, span, sentence, ngrams):
        """
        Apply NegEx labeling functions.
        TODO: Window size is fixed here, choices of 5-8 perform well
        """
        left = get_left_span(span, sentence, window=ngrams)
        right = get_right_span(span, sentence, window=ngrams)

        L = []
        for name in sorted(self.negex.rgxs):
            for cxt in sorted(self.negex.rgxs[name]):
                v = 0
                text = left.text if cxt == 'left' else right.text
                if self.negex.rgxs[name][cxt].search(text):
                    v = self.class_map[name]
                L.append(v)
        return np.array(L)

    def tag(self, document, ngrams=6):
        for i in document.annotations:
            # apply to the following concept targets
            for layer in self.targets:
                if layer not in document.annotations[i]:
                    continue
                for span in document.annotations[i][layer]:
                    L = self._apply_lfs(span, document.sentences[i], ngrams)
                    if L.any() and self.label_reduction == 'mv':
                        y, _ = mode(L[L.nonzero()])
                        span.props['negated'] = y[0]
                    elif L.any() and self.label_reduction == 'or':
                        span.props['negated'] = int(1 in L)