import numpy as np
from functools import partial
from statistics import mode

from trove.contrib.labelers.clinical.helpers import *
from trove.contrib.labelers.clinical.taggers import Tagger
from trove.contrib.labelers.clinical.negex import NegEx

#################################################################################
#
# Family (Subject) Tagger
#
#################################################################################

PATIENT = 1
OTHER   = 2
ABSTAIN = 0

rgx_relatives = re.compile(r'''\b(((grand)*(mother|father)|grand([mp])a)([']*s)*|((parent|(daught|sist|broth)er|son|cousin)([']*s)*))\b''', re.I)


def LF_relative(span):
    """Context includes any familial mention (e.g., mother father)"""
    left = get_left_span(span, span.sentence, window=6)
    right = get_right_span(span, span.sentence, window=6)
    left_trigger = match_regex(rgx_relatives, left)
    right_trigger = match_regex(rgx_relatives, right)
    return OTHER if left_trigger or right_trigger else PATIENT


def LF_header(span, negex):
    """All spans under Family History are assumed to refer to family"""
    rgx = re.compile(r'''(family history[:]*|family hx)\b''', re.I)
    left = get_left_span(span, span.sentence, window=6)
    trigger = match_regex(rgx, left)

    if trigger:
        # check for negation ("no family history")
        neg = match_regex(negex.rgxs['definite']['left'],
                          get_left_span(trigger, window=2))
        return ABSTAIN if neg else OTHER

    if 'section' in span.props:
        header = span.props['section']
        if header and rgx.search(header.text):
            return OTHER
    return ABSTAIN


def LF_social(span):
    rgx_social = re.compile(
        r'''\b(friend(s)*|roomate(s)*|passenger(s)*)\b''', re.I
    )
    left = get_left_span(span, span.sentence, window=6)
    right = get_right_span(span, span.sentence, window=6)
    left_trigger = match_regex(rgx_social, left)
    right_trigger = match_regex(rgx_social, right)
    return OTHER if left_trigger or right_trigger else ABSTAIN


def LF_history_of(span):
    rgx = r'''\bfamily (history of|hx)'''
    text = get_left_span(span, span.sentence, window=6).text
    return OTHER if re.search(rgx, text.strip(), re.I) else ABSTAIN


def LF_ext_family(span):
    rgx = re.compile(r'''\b(spouse|wife|husband)\b''', re.I)
    text = get_left_span(span, span.sentence, window=6).text
    return OTHER if rgx.search(text) else ABSTAIN


def LF_donor(span):
    rgx = r'''\b(donor)\b'''
    text = get_left_span(span, span.sentence, window=6).text
    return OTHER if re.search(rgx, span.sentence.text.strip(), re.I) else ABSTAIN


class FamilyTagger(Tagger):
    """
    Concepts are generally attached to the patient. However, there
    are cases where concepts attach to family members or donors.
    """
    def __init__(self, targets, data_root, label_reduction='or'):
        self.prop_name = 'subject'
        self.targets = targets
        self.label_reduction = label_reduction
        self.negex = NegEx(data_root=data_root)

        self.lfs = [
            LF_relative,
            LF_ext_family,
            LF_social,
            partial(LF_header, negex=self.negex),
            LF_history_of,
            LF_donor
        ]

        self.class_map = {
            1: "patient",
            2: "family/other"
        }

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
                            # break ties
                            y = 2
                        span.props[self.prop_name] = self.class_map[y]

                    # logical or
                    elif L.any() and self.label_reduction == 'or':
                        if 2 in L:
                            span.props[self.prop_name] = self.class_map[2]
                        else:
                            span.props[self.prop_name] = self.class_map[1]
