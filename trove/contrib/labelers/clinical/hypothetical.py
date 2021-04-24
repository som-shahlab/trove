from trove.contrib.labelers.clinical.helpers import get_left_span
from trove.contrib.labelers.clinical.taggers import *
from scipy.stats import mode
import numpy as np

###############################################################################
#
# Hypothetical Tagger
#
###############################################################################

class HypotheticalTagger(Tagger):
    """
    Hypothetical future events. These are discussed in future tense as
    speculative events.
    - "assuming X happens"
    - "recommend X"
    - "chance of X"
    - "Please call if X happens"
    """

    def __init__(self, targets, label_reduction='or'):
        """
        label_reduction:  or|mv
        """
        self.targets = targets
        self.label_reduction = label_reduction

        accept_rgxs = [
            r'''\b(if need be)\b''',
            r'''\b((if|should)\s+(you|she|he|be)|(she|he|you)\s+(might|could|may)\s*(be)*|if)\b''',
            r'''\b((possibility|potential|chance|need) (for|of)|potentially)\b''',
            r'''\b(candidate for|pending)\b''',
            r'''\b(assuming)\s+(you|she|he)\b''',
            r'''(recommendation)\s*[:]''',
            r'''(planned procedure)\s*[:]''',
            r'''\b(evaluated for|upcoming|would benefit from|(undergo|requires) a)\b''',
            r'''\b(please call or return (for|if))\b''',
            r'''\b(discussed|discussion|recommended|recommendation made|proceed with|consider|to undergo|scheduled for)\b'''
        ]

        reject_rgxs = [
            r'''\b((months|years|days)*\s*(postop|post[- ]op|out from))\b''',
            r'''\b((month|year|day)[s]* post)\b''',
            r'''\b((week|month|year)*[s]*\s*status post)\b'''
        ]
        self.header = [f'LF_accept_{i}' for i in range(len(accept_rgxs))]
        self.header += [f'LF_reject_{i}' for i in range(len(reject_rgxs))]

        self.accept_rgxs = [re.compile(rgx, re.I) for rgx in accept_rgxs]
        self.reject_rgxs = [re.compile(rgx, re.I) for rgx in reject_rgxs]

    def _apply_lfs(self, span, sentence, ngrams=10):
        """
        - This only considers LEFT context windows

        """
        left = get_left_span(span, sentence, window=ngrams)

        L = []
        for rgx in self.accept_rgxs:
            v = 1 if rgx.search(left.text) else 0
            L.append(v)
        for rgx in self.accept_rgxs:
            v = 2 if rgx.search(left.text) else 0
            L.append(v)
        return np.array(L)

    def tag(self, document, ngrams=10):
        for i in document.annotations:
            # apply to the following concept targets
            for layer in self.targets:
                if layer not in document.annotations[i]:
                    continue
                for span in document.annotations[i][layer]:
                    L = self._apply_lfs(span, document.sentences[i], ngrams)
                    if L.any() and self.label_reduction == 'mv':
                        y, _ = mode(L[L.nonzero()])
                        span.props['hypothetical'] = y[0]
                    elif L.any() and self.label_reduction == 'or':
                        if int(1 in L):
                            span.props['hypothetical'] = 1