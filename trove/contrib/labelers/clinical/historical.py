import datetime
import numpy as np
from functools import partial
from scipy.stats import mode
from trove.dataloaders.contexts import Span
from trove.contrib.labelers.clinical.helpers import *
from trove.contrib.labelers.clinical.taggers import Tagger


#################################################################################
#
# Historical Tagger
#
#################################################################################

ABSTAIN = 0
POSITIVE = 1
NEGATIVE = 2

def LF_underspecified_date(span):
    """
    The DATETIME markup layer only contains datetimes that are easy to
    normalize with high accuracy at the day-level (e.g., January 11, 2000,
    11-13-1980). Many notes contain additional, but underspecificed dates,
    e.g.,

    - h/o joint replacements in B hips and shoulders from osteonecrosis in 1980
    - LLE DVT in 2000 and LV thrombus discovered in 2001 during admission for MI

    When we have a note-level timestamp, we can heuristically normalize
    underspecified dates.

    """
    doc = span.sentence.document
    doc_ts = doc.props['doctime']
    annotations = doc.annotations[span.sentence.position]
    #dates = annotations['DATETIME'] if 'DATETIME' in annotations else []
    #dates = annotations['TIMEX3'] if 'TIMEX3' in annotations else []

    if not doc_ts:
       return ABSTAIN

    try:
        # year
        for match in re.finditer("(19[0-9]{2}|20[01][0-9])+s*", span.sentence.text, re.I):
            year = int(match.group().strip("s"))
            if doc_ts.year > year:
                return POSITIVE
    except:
        return ABSTAIN

    # month/date
    m = re.search(r'''on ((1[12]|[1-9])[/-](3[01]|[12][0-9]|[1-9]))\b''',
                  span.sentence.text,
                  re.I)
    if m:
        try:
            month, date = map(int, re.split("[/-]", m.group(1)))
            ts = datetime.datetime(doc_ts.year, month, date)
            if doc_ts > ts:
                return POSITIVE
        except:
            return ABSTAIN

    return 0

def LF_in_history_headers(span):
    """
    Spans under these headers are typically historical events.

    """
    reject_headers = {
        'admitting history',
        'clinical history',
        'oncologic history',
        'past medical history',
        'past medical/surgical history',
        'past surgical history'
    }
    accept_headers = {
        'history of present illness'
    }
    annos = span.sentence.document.annotations[span.sentence.position]
    if 'HEADER' not in annos:
        return ABSTAIN
    headers = set([h.text.lower().replace(':', '') for h in annos['HEADER'] if h])
    return POSITIVE if reject_headers.intersection(headers) and not accept_headers.intersection(
        headers) else ABSTAIN

def LF_history_of(span, window=25):
    """Historical mention detection """
    i = span.get_word_start()
    left = " ".join(span.sentence.words[max(0, i - window):i])
    text = f'{left} {span.text}'

    accept_left_rgxs = [
        r'''\b(h/o|hx|history of)\b''',
        r'''\b(s/p|SP|status[- ]post)\b''',
        r'''\b(recent|previous)\b''',
        r'''\b(in the (distant )*past)\b''',
        r'''\b([0-9]{1,2} ((day|week|month|year)[s]*) prior)\b'''

    ]
    reject_left_rgxs = [
        r'''\b(history of present illness|chief complaint|indication)[:]*\b''',
        r'''\b(p/w|present(ed|s) with)\b''',
        r'''\b(new onset)\b'''
    ]

    accept_right_rgxs = [
        r'''\b(in the (distant )*past)\b'''
    ]

    for rgx in reject_left_rgxs:
        if re.search(rgx, text, re.I):
            return NEGATIVE

    for rgx in accept_left_rgxs:
        m = re.search(rgx, text, re.I)
        if m:
            return POSITIVE

    return ABSTAIN


def LF_doctime_complication(span):
    """
    Use explicit date mentions in a sentence as a heuristic for labeling past events.
    """
    return POSITIVE if 'tdelta' in span.props and span.props['tdelta'] < 1 else ABSTAIN


def LF_in_history_of_list(span, targets):
    """
    Two styles of patient history mentions:
     - Statement: 'history of multiple myeloma and multiple prior surgeries'
     - List: 'H/O bilateral hip replacements; MRSA infection; Osteopenia ...'
    """
    for match in re.finditer(r'''\b(h/o|hx|history of)\b''', span.sentence.text):

        # left  = Span(0, span.char_start-1, sentence)
        # right = Span(span.char_end+1, len(sentence.text), sentence)

        i, j = match.span()
        right = Span(j, len(span.sentence.text), span.sentence)

        annos = span.sentence.document.annotations[span.sentence.position]

        concepts = []
        for name in targets: #["DISORDER", "PROCEDURE"]:
            if name not in annos:
                continue
            concepts.extend([s for s in annos[name]])

        dates = [s for s in annos['DATETIME']] if 'DATETIME' in annos else []

        # sentence contains multple dates or concepts
        if len(dates) > 2 or len(concepts) > 4:
            return POSITIVE

        # list of multiple elements
        if len(re.split("[;,]", right.text, re.I)) > 4:
            return POSITIVE

    return ABSTAIN


class HistoricalTagger(Tagger):
    """

    NOTE: We currently use a more restrictive definition of historical that
    PyConText, labeling any event as historical that only place a event in the
    past. For example, events that are ongoing (i.e., imply past occurences
    as well) are not labeled as historical

    - chronic pain
    - recurrent infections
    - increase in X
    etc

    """

    def __init__(self, targets, label_reduction='or'):
        self.prop_name = 'past'
        self.targets = targets
        self.label_reduction = label_reduction

        self.lfs = [
            LF_doctime_complication,
            LF_in_history_headers,
            LF_history_of,
            LF_underspecified_date,
            partial(LF_in_history_of_list, targets=targets)
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
                        y, _ = mode(L[L.nonzero()])
                        span.props[self.prop_name] = y[0]
                    # logical or
                    elif L.any() and self.label_reduction == 'or':
                        if 1 in L:
                            span.props[self.prop_name] = 1
                    # label matrix
                    elif L.any() and self.label_reduction == 'matrix':
                        span.props[self.prop_name] = L
