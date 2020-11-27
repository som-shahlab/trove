# trivial-to-mild
# mild-minimal
#
# slight
# mild-to-minimal
# mild-to-modest
# mild-moderate
# mild-to-moderate
#
# considerable
# substantial
# widespread
#
# jabbing
# sharp-like
# knifelike
# sharp-stabbing
# stabbing
# throbbing
# unbearable
# intense

import re
import numpy as np
#from scipy.stats import mode
from statistics import mode
from .helpers import *
from .taggers import Tagger


ABSTAIN  = 0
SLIGHT   = 1
MODERATE = 2
SEVERE   = 3
UNMARKED = 4

def LF_slight(span):
    rgx = r'''\b((slight|minimal)(ly)*|trace|minor|trivial|little|partial|min)\b'''
    text = get_left_span(span, span.sentence, window=6).text
    return SLIGHT if re.search(rgx, text, re.I) else ABSTAIN


def LF_moderate(span):
    rgx = r'''((moderate|mild)(ly)*|large)'''
    text = get_left_span(span, span.sentence, window=6).text
    return MODERATE if re.search(rgx, text, re.I) else ABSTAIN


def LF_severe(span):
    rgx = r'''(sharp|knife-like|significant|extensive|extreme|(marked|severe)(ly)*|severity)'''
    text = get_left_span(span, span.sentence, window=6).text
    return SEVERE if re.search(rgx, text, re.I) else ABSTAIN


class SeverityTagger(Tagger):

    def __init__(self, targets, data_root, label_reduction='mv'):
        """
        label_reduction:  or|mv
        """
        self.prop_name = 'severity'
        self.targets = targets
        self.label_reduction = label_reduction

        self.class_map = {
            1: 'slight',
            2: 'moderate',
            3: 'severe',
            4: 'NULL'
        }

        self.lfs = [
            LF_slight,
            LF_moderate,
            LF_severe
        ]

    def _apply_lfs(self, span):
        """ Apply labeling functions. """
        return np.array([lf(span) for lf in self.lfs])

    def tag(self, document, **kwargs):
        for i in document.annotations:
            # apply to the following concept targets
            for layer in self.targets:
                if layer not in document.annotations[i]:
                    continue
                for span in document.annotations[i][layer]:
                    L = self._apply_lfs(span)
                    # majority vote
                    if L.any() and self.label_reduction == 'mv':
                        try:
                            y = mode(L[L.nonzero()])
                        except:
                            y = 4 # break ties
                        span.props[self.prop_name] = self.class_map[y]

                    # label matrix
                    elif L.any() and self.label_reduction == 'matrix':
                        span.props[self.prop_name] = L