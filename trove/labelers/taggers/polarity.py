import re
import numpy as np
from statistics import mode
from .helpers import *
from .taggers import Tagger
from functools import partial
from .negex import NegEx

ABSTAIN = 0
NEGATED = 1
NON_NEGATED = 2

# not associated with| No improvement in
pseudo_negation_rgx = re.compile(r'''(limited to|rule out)''', re.I)
#pseudo_negation_left = re.compile(r'''(not associated with|no improvement in)''', re.I)


def get_containing_span(span):
    # add end padding to make computing span simpler
    text = span.sentence.text + ' '
    start, end = span.char_start, span.char_end

    i = start - 1
    for i in range(start - 1, 0, -1):
        if text[i] == ' ':
            break

    j = end
    for j in range(end, len(text), 1):
        if text[j] == ' ':
            break

    return Span(char_start=i + 1, char_end=j - 1, sentence=span.sentence)


def LF_plus_minus_prefix(span):
    """
    Clinical text uses some shorthand for positive/negative
    presence of conditions, e.g.,
        Chest: CTAB -w/r/r
    """
    # TODO - fix this span bug
    if len(span.text) == 0:
        return ABSTAIN

    reject_sections = [
        re.compile(r'''past medical history''', re.I)
    ]
    for rgx in reject_sections:
        if rgx.search(span.sentence.text):
            return ABSTAIN

    cspan = get_containing_span(span)
    modifier = cspan.text[0]
    if re.search(r'''^[-=]([A-Z]+|[A-Za-z][/])\b''', cspan.text):
        return NEGATED

    elif modifier in ['+']:
        return NON_NEGATED
    return ABSTAIN


def LF_definite_left_0(span, negex):
    left = get_left_span(span, span.sentence, window=6)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)
    return NEGATED if dist == 0 else ABSTAIN


def LF_definite_left_1_3(span, negex):
    left = get_left_span(span, span.sentence, window=6)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)

    right = get_right_span(trigger, window=2)
    if pseudo_negation.search(right.text):
        return ABSTAIN

    return NEGATED if (dist >= 1 and dist < 4) else ABSTAIN


def LF_definite_left_4_6(span, negex):
    left = get_left_span(span, span.sentence, window=6)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)

    right = get_right_span(trigger, window=2)
    if pseudo_negation.search(right.text):
        return ABSTAIN

    return NEGATED if (dist >= 4 and dist <= 6) else ABSTAIN


def LF_definite_left_7_10(span, negex):
    left = get_left_span(span, span.sentence)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)

    right = get_right_span(trigger, window=2)
    if pseudo_negation.search(right.text):
        return ABSTAIN

    return NEGATED if (dist >= 7 and dist <= 10) else ABSTAIN


def LF_probable_left_0(span, negex):
    left = get_left_span(span, span.sentence, window=6)
    trigger = match_regex(negex.rgxs['probable']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)
    return NEGATED if dist == 0 else ABSTAIN


def LF_probable_left_1_3(span, negex):
    left = get_left_span(span, span.sentence, window=6)
    trigger = match_regex(negex.rgxs['probable']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)
    right = get_right_span(trigger, window=2)
    if pseudo_negation.search(right.text):
        return ABSTAIN
    return NEGATED if (dist >= 1 and dist < 4) else ABSTAIN


def LF_probable_left_4_6(span, negex):
    left = get_left_span(span, span.sentence, window=6)
    trigger = match_regex(negex.rgxs['probable']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)
    right = get_right_span(trigger, window=2)
    if pseudo_negation.search(right.text):
        return ABSTAIN
    return NEGATED if (dist >= 4 and dist <= 6) else ABSTAIN


def LF_definite_left(span, negex):
    text = get_left_span(span, span.sentence, window=6).text
    rgx = negex.rgxs['definite']['left']
    if pseudo_negation(span):
        return ABSTAIN
    return NEGATED if rgx.search(text) else ABSTAIN


def LF_definite_right(span, negex):
    text = get_right_span(span, span.sentence, window=6).text
    rgx = negex.rgxs['definite']['right']
    if pseudo_negation(span):
        return ABSTAIN
    return NEGATED if rgx.search(text) else ABSTAIN


def LF_probable_left(span, negex):
    text = get_left_span(span, span.sentence, window=6).text
    rgx = negex.rgxs['probable']['left']
    if pseudo_negation(span):
        return ABSTAIN
    return NEGATED if rgx.search(text) else ABSTAIN


def LF_probable_right(span, negex):
    text = get_right_span(span, span.sentence, window=6).text
    rgx = negex.rgxs['probable']['right']
    if pseudo_negation(span):
        return ABSTAIN
    return NEGATED if rgx.search(text) else ABSTAIN


def LF_pseudo_left(span, negex):
    text = get_left_span(span, span.sentence, window=6).text
    rgx = negex.rgxs['pseudo']['left']
    return NON_NEGATED if rgx.search(text) else ABSTAIN


def LF_left_context(span):
    left = get_left_span(span, span.sentence, window=6)

    # negated mentions
    neg_rgxes = [
        r'''\b(no|did not have|neg(ative)* for) (mild|slight|minimal|severe|moderate|extensive|marked|extreme|significant|progressive)\b''',
        r'''\b(no|did not have|neg(ative)* for) (known|evidence of|evidence)\b'''
    ]
    neg_rgxes = [re.compile(rgx, re.I) for rgx in neg_rgxes]

    for rgx in neg_rgxes:
        trigger = match_regex(rgx, left)
        if trigger and token_distance(trigger, span) <= 2:
            return NEGATED

    # positive mentions
    pos_regxes = [
        r'''(cannot exclude|does not become|may not|possible|evaluate for|suggests)''',  # hedged
        r'''(mild|minimal|severe|moderate|extensive|coarse|marked|extreme|significant|trivial|progressive|slight)(ly)*''',
        # severity
        r'''(diagnosed with|known to have|known|non-specific|presented|secondary to|treated for|acute onset|improving|improved|improvement|involvement|resolved|consistent with|showed|presumed|suspicious for|check for|revealed|new onset|were noted|found to be|demonstrate(d)*)''',
        # present now
        r'''\b((in|de)creas(e|ed|ing)|up|down)\b''',  # LF_change_words_left
        r'''(s/p|status[- ]post)''',

    ]
    pos_regxes = [re.compile(rgx, re.I) for rgx in pos_regxes]

    for rgx in pos_regxes:
        if rgx.search(span.text):
            return NON_NEGATED
    return ABSTAIN


def LF_right_context(span):
    text = get_right_span(span, span.sentence, window=6).text
    regxes = [
        r'''((?<!no )(mild|minimal|severe|moderate|extensive|coarse|marked|extreme|significant|trivial|progressive|slight)(ly)*)''',
        # LF_severity_right
        r'''(was (found to have|impaired|relieved|stable)|is present|withdrawal|of the)'''  # LF_present_now_right
    ]
    regxes = [re.compile(rgx, re.I) for rgx in regxes]
    for rgx in regxes:
        if rgx.search(text):
            return NON_NEGATED
    return ABSTAIN


def LF_temporal_left(span):
    left = get_left_span(span, window=100)
    rgx = re.compile(
        r'(no|(does|has) not|not had|without|denies) (history of|prior|chronic|residual|occasional|restarted|post-surgical changes|again noted|immediate(ly)*|remained on)',
        re.I)
    match = rgx.search(left.text)
    if not match:
        return ABSTAIN
    if re.search(r'''(no|(does|has) not|not had|without|denies)\b''', match.group(), re.I):
        return NEGATED
    else:
        return NON_NEGATED


def LF_short_sentence(span):
    """A sentence mostly consisting of the target span and no negation words."""
    rgx = re.compile(
        r'(no|not|never|cannot|negative for|negative|neg|absent|ruled out|without|absence of|den(y|ied|ies))', re.I)
    v = len(span.sentence.words) < 5
    v &= not rgx.search(span.sentence.text)
    return NON_NEGATED if v else ABSTAIN


def LF_no_negation_terms(span):
    """No negation words or punctuation are found anywhere in the sentence."""
    rgx = re.compile(
        r'''\b(no|w[/]o|[(][-][)]|not|non|none|free|never|cannot|negative for|negative|neg|absent|ruled out|without|absence of|den(y|ied|ies))\b''',
        re.I)
    v = not rgx.search(span.sentence.text)
    v &= not re.search(r'''[-]''', span.sentence.text)
    return NON_NEGATED if v else ABSTAIN


def LF_header(span):
    rgx = re.compile(
        r'^(admitting diagnosis|chief complaint|discharge diagnosis|past medical history|history of present illness|indication)[:]*',
        re.I)
    v = rgx.search(span.sentence.text.strip()) is not None
    right = get_right_span(span, window=1).text
    v |= right == ':'
    return NON_NEGATED if v else ABSTAIN


def LF_head_word(span):
    rgx = re.compile(
        r'''^(no|not|never|cannot|negative for|negative|neg|absent|ruled out|without|absence of|den(y|ied|ies))\b''',
        re.I)
    left = get_left_span(span, span.sentence)
    n = len(left.get_attrib_span('words'))
    return NEGATED if rgx.search(left.text.strip()) else ABSTAIN


def LF_terminator_word_left(span):
    trigger_rgx = re.compile(
        r'''(no|not|never|cannot|negative for|negative|neg|absent|ruled out|without|absence of|den(y|ied|ies))\b''',
        re.I)
    rgx = re.compile(r'''\b(but|after|post|prior|before|during|rather than)\b|\n|[:;]''', re.I)
    # find closest trigger word
    text = get_left_span(span, span.sentence, window=6).text
    matches = [m for m in trigger_rgx.finditer(span.sentence.text) if m.span()[-1] < span.char_start]
    start = len(text)
    trigger = None
    for m in matches:
        if trigger:
            d1 = start - trigger.span()[-1]
            d2 = start - m.span()[-1]
            if d2 < d1:
                trigger = m
        else:
            trigger = m
    # does a terminator word occur between?
    if trigger:
        window = text[trigger.span()[-1]:]
        return NON_NEGATED if rgx.search(window) else ABSTAIN
    return ABSTAIN


def LF_pseudo_left_exp(span, negex):
    pseudo_rgx = re.compile(
        r'''\b(exclude|improvement of|performed|be quantified|not limited to|be adequately assessed|do not indicate|significant change)\b''',
        re.I)
    rgx = negex.rgxs['definite']['left']
    text = get_left_span(span, span.sentence, window=6).text
    return NON_NEGATED if rgx.search(text) and pseudo_rgx.search(text) else ABSTAIN


def LF_pseudo_left_expanded(span, negex):
    pseudo_rgxs = [
        r'''\b((significant )*(change[s]* in|improvement))\b''',
        r'''\b((in|de)creas(e|ed|ing)|up|down)\b'''
    ]
    left = get_left_span(span)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    if not trigger or token_distance(trigger, span) > 20:
        return ABSTAIN

    btw = get_between_span(span, trigger)
    if not btw:
        return ABSTAIN

    for rgx in pseudo_rgxs:
        if re.search(rgx, btw.text, re.I):
            return NON_NEGATED

    return ABSTAIN


def LF_denies_list(span):
    """ Patient denies X,Y,Z. """
    rgx = re.compile(r'''\b(den(ying|y|ies|ied))\b''', re.I)
    left = get_left_span(span, window=100)
    trigger = match_regex(rgx, left)
    if not trigger:
        return ABSTAIN

    btw = get_between_span(trigger, span)
    if not btw:
        return ABSTAIN

    n = len(re.findall(r'''[,;/]''', btw.text))
    return NEGATED if n >= 1 else ABSTAIN


def LF_verb_left(span):
    left = get_left_span(span, span.sentence, window=50).text
    rgx = r'''((no|not|(den(y|ies|ied|ying))) )*(\w+ ){1,}(is a|is|will be|are)$'''
    m = re.search(rgx, left, re.I)
    return NON_NEGATED if m and not re.search(r'''\b(no|not|den(y|ies|ied|ying))\b''', m.group(), re.I) else ABSTAIN


def LF_positive_left(span):
    left = get_left_span(span, window=10)
    rgx = r'''\b(positive for|suggestive of|due to|shows)\b'''
    trigger = match_regex(rgx, left)
    if not trigger:
        return ABSTAIN
    left = get_left_span(trigger, window=50)
    m = re.search(r'''\b(no|not)\b''', left.text, re.I)
    return NON_NEGATED if not re.search(r'''\b(no|not)\b''', left.text, re.I) else ABSTAIN


def LF_definite_left_list(span, negex):
    left = get_left_span(span, span.sentence, window=100)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    if not trigger:
        return ABSTAIN
    dist = token_distance(trigger, span)
    btw = get_between_span(trigger, span)

    right = get_right_span(trigger, window=2)
    if pseudo_negation_rgx.search(right.text):
        return ABSTAIN

    return NEGATED if dist <= 10 and btw and len(re.findall(r'''[,;]''', btw.text)) > 1 else ABSTAIN


def LF_left_punct(span):
    cspan = get_containing_span(span)
    left = get_left_span(cspan, span.sentence, window=1)
    if left.text == '+':
        return NON_NEGATED
    return ABSTAIN


def LF_definite_right_expanded(span):
    right = get_right_span(span, span.sentence, window=6)
    rgx = r'''\b(not present|no evidence|is (absent|not seen)|ruled out|were negative)\b'''
    trigger = match_regex(rgx, right)
    if trigger and token_distance(trigger, span) <= 1:
        return NEGATED
    return ABSTAIN


def LF_header_break_negation(span, negex):
    left = get_left_span(span)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    if not trigger:
        return ABSTAIN
    btw = get_between_span(trigger, span)
    if not btw:
        return ABSTAIN
    rgx = r'''\s{2,}((?:(?:[A-Z][A-Za-z]+\s){1,4}(?:[A-Za-z]+))[:])'''
    return NON_NEGATED if re.search(rgx, btw.text) else ABSTAIN


def LF_pseudo_negation_exclusion(span):
    left_rgx = r'''(inadequate\s+to|does\s+not|cannot|can't)\s+exclude'''
    right_rgx = r'''(cannot\s+be|not\s+be|doesn't|not|to)\s+exclude[d]*'''

    left = get_left_span(span)
    trigger = match_regex(left_rgx, left)
    if trigger and token_distance(trigger, span) <= 3:
        return NON_NEGATED

    right = get_right_span(span)
    trigger = match_regex(right_rgx, right)
    if trigger and token_distance(trigger, span) <= 3:
        return NON_NEGATED

    return ABSTAIN


def LF_pseudo_negation_rule_out(span):
    left_rgx = r'''(cannot|does not|doesn't) rule[s]* out'''
    left = get_left_span(span)
    trigger = match_regex(left_rgx, left)
    if not trigger or token_distance(trigger, span) > 5:
        return ABSTAIN
    return NON_NEGATED if re.search(r'''(cannot|does not|doesn't)''',
                                    trigger.text, re.I) else NEGATED

def LF_none(span):
    right_rgx = r'''\bnone\b'''
    right = get_right_span(span)
    trigger = match_regex(right_rgx, right)
    return NEGATED if trigger and token_distance(trigger, span) <= 2 else ABSTAIN


def pseudo_negation(span):
    exclusion = [
        LF_pseudo_negation_exclusion,
        LF_pseudo_negation_rule_out
    ]
    for lf in exclusion:
        if lf(span) == NON_NEGATED:
            return True
    return False


class PolarityTagger(Tagger):

    def __init__(self, targets, data_root, label_reduction='mv'):
        """
        label_reduction:  or|mv
        """
        self.prop_name = 'polarity'
        self.targets = targets
        self.negex = NegEx(data_root=data_root)
        self.label_reduction = label_reduction

        self.class_map = {
            1: 'negated',
            2: 'affirmative'
        }

        self.lfs = [
            partial(LF_definite_left, negex=self.negex),
            partial(LF_probable_left, negex=self.negex),
            partial(LF_definite_right, negex=self.negex),
            partial(LF_probable_right, negex=self.negex),
            partial(LF_pseudo_left, negex=self.negex),

            LF_head_word,
            LF_short_sentence,
            LF_no_negation_terms,
            LF_plus_minus_prefix,
            LF_left_punct,

            LF_left_context,
            LF_right_context,

            LF_temporal_left,
            LF_denies_list,
            LF_verb_left,
            LF_positive_left,
            partial(LF_definite_left_list, negex=self.negex),
            partial(LF_header_break_negation, negex=self.negex),

            LF_pseudo_negation_exclusion,
            LF_pseudo_negation_rule_out,
            LF_none
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
                            y = 2 # break ties
                        span.props[self.prop_name] = self.class_map[y]

                    # logical or
                    elif L.any() and self.label_reduction == 'or':
                        if 1 in L:
                            span.props[self.prop_name] = 1

                    # label matrix
                    elif L.any() and self.label_reduction == 'matrix':
                        span.props[self.prop_name] = L