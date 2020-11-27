import re
import datetime
import numpy as np
from collections import defaultdict
from scipy.stats import mode
from .negex import NegEx
from trove.data.dataloaders.contexts import Span


def token_distance(a,b):
    a,b = sorted([a,b], key=lambda x:x.char_start, reverse=0)
    i,j = a.get_word_end() + 1, b.get_word_start()
    return j - i


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


def overlaps(a, b):
    a_start = a.get_attrib_tokens('abs_char_offsets')[0]
    b_start = b.get_attrib_tokens('abs_char_offsets')[0]
    a_end = a_start + len(a.text)
    b_end = b_start + len(b.text)
    v = a_start >= b_start and a_start <= b_end
    return v or (a_end >= b_start and a_end <= b_end)


def get_left_span(span, sentence=None, window=None):
    """Get window words to the left of span"""
    sentence = sentence if sentence else span.sentence
    j = span.char_to_word_index(span.char_start)
    i = max(j - window, 0) if window else 0
    if i == j == 0:
        return Span(char_start=0, char_end=-1, sentence=sentence)
    start, end = sentence.char_offsets[i], sentence.char_offsets[j-1] + len(sentence.words[j-1]) - 1
    return Span(char_start=start, char_end=end, sentence=sentence)


def get_right_span(span, sentence=None, window=None):
    """Get window words to the right of span"""
    sentence = sentence if sentence else span.sentence
    i = span.get_word_end() + 1
    j = min(i + window, len(sentence.words)) if window else len(sentence.words)
    if i == j:
        return Span(char_start=len(sentence.text), char_end=len(sentence.text), sentence=sentence)
    start, end = sentence.char_offsets[i], sentence.char_offsets[j-1] + len(sentence.words[j-1]) - 1
    return Span(char_start=start, char_end=end, sentence=sentence)


def get_between_span(a, b):
    a, b = sorted([a, b], key=lambda x: x.char_start, reverse=0)
    i, j = a.get_word_end() + 1, b.get_word_start()
    offsets = a.sentence.char_offsets[i:j]
    words = a.sentence.words[i:j]
    if not words:
        return None
    return Span(char_start=offsets[0], char_end=offsets[-1] + len(words[-1]) - 1, sentence=a.sentence)


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


def retokenize(s, split_on=r'''([/-])'''):
    """
    Apply secondary tokenization rule, e.g., split on hyphens or forward slashes.
    """
    words, offsets = [], []
    #for w, i in zip(s.words, s.offsets):
    for w, i in zip(s.words, s.char_offsets):

        tokens = [t for t in re.split(split_on, w) if t.strip()]
        if len(tokens) == 1:
            words.append(w)
            offsets.append(i)
        else:
            offset = i
            for j, t in enumerate(tokens):
                words.append(t)
                offsets.append(offset)
                offset += len(t)
    return words, offsets


class Ngrams(object):
    def __init__(self, n_max=5, split_on=None):
        self.max_ngrams = n_max
        self.split_on = split_on

    def apply(self, s):
        # covert to source char offsets
        text = get_text(s.words, s.char_offsets)

        # apply alternate tokenization
        if self.split_on:
            words, char_offsets = retokenize(s, self.split_on)
        else:
            words, char_offsets = s.words, s.char_offsets

        matches = []
        for i in range(0, len(words)):
            match = None
            start = char_offsets[i]
            # ignore leading whitespace
            if not words[i].strip():
                continue
            for j in range(i + 1, min(i + self.max_ngrams + 1, len(words) + 1)):
                # ignore trailing whitespace
                if not words[j - 1].strip():
                    continue
                end = char_offsets[j - 1] + len(words[j - 1])
                yield Span(start, end - 1, s)


def longest_matches(matches):
    # sort on length then end char
    matches = sorted(matches, key=lambda x: len(x.text), reverse=1)
    matches = sorted(matches, key=lambda x: x.char_end, reverse=1)

    f_matches = []
    curr = None
    for m in matches:
        if curr is None:
            curr = m
            continue
        i, j = m.char_start, m.char_end
        if (i >= curr.char_start and i <= curr.char_end) and (j >= curr.char_start and j <= curr.char_end):
            pass
        else:
            f_matches.append(curr)
            curr = m
    if curr:
        f_matches.append(curr)

    return f_matches


def dict_matcher(sentence,
                 ngrams,
                 dictionaries,
                 min_length=2,
                 stopwords={},
                 longest_match_only=True,
                 ignore_whitespace=True):

    matches = defaultdict(list)
    for span in ngrams.apply(sentence):
        # ignore whitespace when matching dictionary terms
        text = span.text
        if ignore_whitespace:
            text = re.sub(r'''\s{2,}|\n{1,}''', ' ', span.text).strip()

        # search for matches in all dictionaries
        for name in dictionaries:
            if len(text) < min_length or text.lower() in stopwords:
                continue
            if text.lower() in dictionaries[name] or text in dictionaries[name]:
                matches[name].append(span)

    if longest_match_only:
        for name in matches:
            if matches[name]:
                matches[name] = longest_matches(matches[name])

    return matches


###############################################################################
#
# Tagger
#
###############################################################################

class Tagger(object):
    """
    """
    def __init__(self, min_length=2):
        self.min_length = min_length

    def _matches(self, matcher, doc, ngrams, **kwargs):
        """
        Return all matches found in a sentence
        """
        for sent in doc.sentences:
            for m in matcher.apply(ngrams.apply(sent)):
                if len(m.get_span()) > self.min_length:
                    yield (sent.position, m)

    def tag(self, documents, ngrams=10, stopwords=[]):
        raise NotImplementedError()


###############################################################################
#
# Reset All Annotations
#
###############################################################################

class ResetTags(Tagger):
    """
    Clear all document annotations
    """
    def tag(self, document, ngrams=6, stopwords=[]):
        document.annotations = {i:{} for i in range(len(document.sentences))}

###############################################################################
#
# UMLS Tagger 
#
###############################################################################

class UMLSTagger(Tagger):
    """
    Given a UMLS object and a set mapping of class->semantic type,
    generate string matches

    """
    def __init__(self,
                 umls,
                 concepts,
                 min_length=2,
                 longest_match_only=True,
                 stopwords={},
                 custom_terms={}):

        self.umls = umls
        self.longest_match_only = longest_match_only
        self.min_length = min_length
        self.stopwords = stopwords

        self.concept_map = {name: [i.lower().replace(" ", "_") for i in concepts[name]] for name in concepts}
        self.dictionaries = self._build_dictionaries()

        # add custom terms
        for name in self.dictionaries:
            if name in custom_terms:
                self.dictionaries[name].update(dict.fromkeys(custom_terms[name]))

    def _build_dictionaries(self):
        """
        Collapse UMLS semantic types into groups, as defined by `concept_map`.
        """
        dictionaries = {}

        # merge dictionaries
        for name in self.concept_map:
            dictionaries[name] = {}
            for sab, sty in self.umls.dictionary:
                if sty not in self.concept_map[name]:
                    continue
                dictionaries[name].update(dict.fromkeys(self.umls.dictionary[(sab, sty)]))
            print(name, len(dictionaries[name]))

        return dictionaries

    def tag(self, document, ngrams=5):
        """
        """
        candgen = Ngrams(n_max=ngrams)
        for sent in document.sentences:
            m = dict_matcher(sent, candgen, self.dictionaries, min_length=self.min_length, stopwords=self.stopwords)
            if m:
                document.annotations[sent.position].update(dict(m))

###############################################################################
#
# Dictionary Tagger
#
###############################################################################

class DictionaryTagger(Tagger):
    """
    """
    def __init__(self,
                 dictionaries,
                 min_length=2,
                 longest_match_only=True,
                 stopwords={},
                 split_on=None):

        self.dictionaries = dictionaries
        self.longest_match_only = longest_match_only
        self.min_length = min_length
        self.stopwords = stopwords
        self.split_on = split_on

    def tag(self, document, ngrams=5):
        """
        """
        candgen = Ngrams(n_max=ngrams, split_on=self.split_on)
        for sent in document.sentences:
            m = dict_matcher(sent, candgen, self.dictionaries, min_length=self.min_length, stopwords=self.stopwords)
            if m:
                document.annotations[sent.position].update(dict(m))



###############################################################################
#
# DocTime Tagger
#
###############################################################################

class DocTimeTagger(Tagger):
    """
    Estimte document timestamp. Use either:
    1: Explicit note sign date
    2: Most recent unambiguous TIMEX mention
    """

    def __init__(self, targets=['DATETIME', 'HEADER'], field='T'):
        self.targets = targets
        self.field = field

    def tag(self, document):

        max_date, sign_dates = None, []
        for i in range(len(document.annotations)):
            header = document.annotations[i]['HEADER'][0] if 'HEADER' in document.annotations[i] else None
            timexs = document.annotations[i]['DATETIME'] if 'DATETIME' in document.annotations[i] else []
            ts = [ts.normalized for ts in timexs if ts.normalized]
            if ts:
                max_date = max([max_date] + ts) if max_date else max(ts)
            if ts and header and re.search("^\s*{}[:]".format(self.field), header.text):
                sign_dates.extend(ts)

        # defer to timestamps in the target field section
        if sign_dates:
            document.props['doctime'] = max(sign_dates)
        # select max from all datetimes
        elif max_date:
            document.props['doctime'] = max_date
        else:
            document.props['doctime'] = None


###############################################################################
#
# Time Delta Tagger
#
###############################################################################

class TimeDeltaTagger(Tagger):
    """
    =======================
    CONCEPT TYPE: Relation
    =======================

    Link spans to their nearest (in token distance) normalized date mention. For example:

    INPUTS: DOCTIME:  12/15/2017
            SPAN:     hip implant
            SENTENCE: On 11/13/2017, the patient came into the clinic with an infected hip implant.
    OUTPUT: (hip implant, -32 days)

    CAVEATS:
    - We restrict to datetime mentions found in the same sentence as our target span.
    - We ignore spans that occur within HEADER mentions
    - Relying on only the nearest DATETIME mention fails in many cases. Consider the examples:

        + Chief Complaint: 60 year old white female s/p CABG 2016-01-05 with fever and sternal wound drainage.

      Coronary artery bypass grafting (CABG) was done on 2016-01-05,
      but 'fever' and 'sternal wound drainage' are current conditions.

        + Since the previous tracing of 2015-03-17 sinus bradycardia rate is slower.

      Current slower sinus bradycardia is being compared to a previous tracing on 2015-03-17.

    TODOs:
    - Create interface for ranked list (by token distance) of all datetimes given a span 
      to allow for more flexible LF design.
    
    """
    def __init__(self, targets):
        self.targets = targets

    def _apply_lfs(self, span):
        # TODO: Implement labeling functions for this task.
        pass

    def tag(self, document):
        doc_ts = document.props['doctime']
        if doc_ts is None:
            print(f"no pops[doctime] for {document.name}")
        else:
            for i, sent in enumerate(document.sentences):
                # get sentence datetime mentions
                if 'TIMEX3' not in document.annotations[i]:
                    continue

                hdr = document.annotations[i]['HEADER']
                sent_ts = [ts for ts in document.annotations[i]['TIMEX3'] if ts.normalized]
                if not sent_ts:
                    continue

                for name in set(document.annotations[i]).intersection(self.targets):
                    for span in document.annotations[i][name]:
                        # ignore spans that overlap HEADER entities
                        if hdr[0] and overlaps(span, hdr[0]):
                            continue
                        # select closest DATETIME mention
                        dists = [token_distance(span, ts) for ts in sent_ts]
                        dists = sorted(zip(dists, sent_ts), key=lambda x: x[0], reverse=0)
                        tdelta = dists[0][-1]
                        span.props['tdelta'] = (tdelta.normalized - doc_ts).days
                        span.props['timex']  = tdelta.normalized
                        span.props['timex_span'] = tdelta


###############################################################################
#
# Negation Tagger
#
###############################################################################

class NegExTagger(Tagger):
    """
    Map each negex regex/context type to a labeling function.

    """

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
        Apply labeling functions.

        TODO: Window size is fixed here. We can either:
        1: Tune window value (5-8 seems reasonable)
        2: Use an exponentially weighted random value *per call*.
           In expectation, this is equiv. to using multiple window
           choices in terms of est. LF params.
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
                        span.props['neg'] = y[0]
                    elif L.any() and self.label_reduction == 'or':
                        span.props['neg'] = int(1 in L)

###############################################################################
#
# Laterality Tagger
#
###############################################################################

class LateralityTagger(Tagger):
    """
    Right/Left/Bilateral spatial modifier.

    """

    def __init__(self, targets):
        self.labels = {'LEFT': 1, 'RIGHT': 2, 'BILATERAL': 3}
        self.targets = targets

    def _get_normed_laterality(self, t):
        laterality_map = {
            'L': ['left', 'lt', 'l', 'left-sided', 'left sided', 'l-sided', 'l sided'],
            'R': ['right', 'rt', 'r', 'right-sided', 'right sided', 'r-sided', 'rt sided'],
            'B': ['bilateral', 'r/l', 'b/l', 'bilat']
        }
        t = t.text.lower() if t else None
        for grp in laterality_map:
            if t in laterality_map[grp]:
                return grp
        return None

    def _get_laterality(self, span, sentence=None, window=3):
        """
        Extract closest laterality mention and normalize to a canonical format
        """
        laterality_rgx = [
            r'''\b(bilat(eral)*|r/l|b/l)\b''',
            r'''\b((left|right)[- ]*side[d]*|\( (left|right) \)|(left|right)|\( [lr] \)|(lt|rt)[.]*|[lr])\b'''
        ]
        laterality_rgx = "|".join(laterality_rgx)
        sent = span.get_parent() if not sentence else sentence

        # laterality mentioned in the entity?
        for match in re.finditer(laterality_rgx, span.text, re.I):
            if match:
                start, end = match.span()
                return Span(char_start=span.char_start + start, char_end=span.char_start + end - 1, sentence=sent)

        # check context windows
        left = get_left_span(span, sent, window=window)
        right = get_right_span(span, sent, window=window)

        # left window
        matches = []
        for match in re.finditer(laterality_rgx, left.get_span(), re.I):
            start, end = match.span()
            ts = Span(char_start=left.char_start + start, char_end=left.char_start + end - 1, sentence=sent)
            dist = span.char_start - ts.char_end
            matches.append((dist, ts))
        # return closest match
        if matches:
            for dist, tspan in sorted(matches, reverse=0):
                return tspan

        # right window
        # for match in re.finditer(laterality_rgx, right.get_span(), re.I):
        #     start, end = match.span()
        #     ts = Span(char_start=right.char_start+start, char_end=right.char_start+end-1, sentence=sent)
        #     dist = ts.char_start - span.char_end
        #     matches.append((dist, ts))
        # if matches:
        #     for dist,tspan in sorted(matches, reverse=0):
        #         return tspan

        return None

    def tag(self, document, ngrams=2):
        for i in document.annotations:
            # apply to the following concept targets
            for layer in self.targets:
                if layer not in document.annotations[i]:
                    continue
                for span in document.annotations[i][layer]:
                    laterality = self._get_laterality(span, document.sentences[i], window=ngrams)
                    if laterality:
                        span.props['lat'] = self._get_normed_laterality(laterality)


###############################################################################
#
# Hypothetical Tagger
#
###############################################################################

class HypotheticalTagger(Tagger):
    """
    Hypothetical future events. These are discussed in future tense as speculative events.
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
            r"\b(if need be)\b",
            r"\b((if|should)\s+(you|she|he|be)|(she|he|you)\s+(might|could|may)\s*(be)*|if)\b",
            r"\b((possibility|potential|chance|need) (for|of)|potentially)\b",
            r"\b(candidate for|pending)\b",
            r"\b(assuming)\s+(you|she|he)\b",
            r"(recommendation)\s*[:]",
            r"(planned procedure)\s*[:]",
            r"\b(evaluated for|upcoming|would benefit from|(undergo|requires) a)\b",
            r'''\b(please call or return (for|if))\b''',
            r"\b(discussed|discussion|recommended|recommendation made|proceed with|consider|to undergo|scheduled for)\b"
        ]

        reject_rgxs = [
            r"\b((months|years|days)*\s*(postop|post[- ]op|out from))\b",
            r"\b((month|year|day)[s]* post)\b",
            r"\b((week|month|year)*[s]*\s*status post)\b"
        ]
        self.header = [f'LF_accept_{i}' for i in range(len(accept_rgxs))]
        self.header += [f'LF_reject_{i}' for i in range(len(reject_rgxs))]

        self.accept_rgxs = [re.compile(rgx, re.I) for rgx in accept_rgxs]
        self.reject_rgxs = [re.compile(rgx, re.I) for rgx in reject_rgxs]

    def _apply_lfs(self, span, sentence, ngrams=10):
        """
        TODO:
        - This only considers LEFT context windows
        - Incorporate ConText heuristics

        """
        left = get_left_span(span, sentence, window=ngrams)
        # right = get_right_span(span, sentence, window=5)

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
                        span.props['hypo'] = y[0]
                    elif L.any() and self.label_reduction == 'or':
                        span.props['hypo'] = int(1 in L)

###############################################################################
#
# Section Header Tagger
#
###############################################################################

class SectionHeaderTagger(Tagger):
    """
    Identify possible section headers such as:
        HPI:
        HOSPITAL COURSE:

    TODOs: 
    - Remove any entity spans that are subsequences of HEADER spans
    - Allow for dictionary of common headers
    
    """
    def __init__(self, max_token_len=6):
        self.max_token_len = max_token_len
        self._init()

    def _init(self):
        """
        Regular expression for detecting section headers.
        """
        #NOTE: slightly higher recall, but slower!
        #rgx = r'''^[\s\n0-9.)]*((([A-Za-z#.,/\\-]+|24hrs)\s*){1,''' + str(self.max_token_len) + '''}[:])'''
        #rgx = r'''^[\s\n]+((([A-Za-z#.,/\\-]+|24hrs)\s*){1,''' + str(self.max_token_len) + '''}[:])'''
        rgx = '(?:^[\s\n]*|[\n])((?:(?:[A-Za-z#.,/\\-]+|24hrs)\s{0,1}){1,' + str(self.max_token_len) + '}[:])'

        self.matchers = {"HEADER": re.compile(rgx, re.I)}

    def _matches(self, rgx, doc, ngrams, group=0):
        """
        For each sentence, return all matches for the provided regex pattern.
        """
        text = ''
        for i, sent in enumerate(doc.sentences):
            for match in rgx.finditer(sent.text):
            #for match in re.finditer(rgx, sent.text, re.I):
                span = match.span(group)
                start, end = span
                tspan = Span(char_start=start, char_end=end - 1, sentence=sent)
                yield (i, tspan)

    def tag(self, document, ngrams=6, stopwords=[]):
        """ """
        candgen = Ngrams(n_max=ngrams)
        matches = defaultdict(list)
        for sidx, match in self._matches(self.matchers["HEADER"], document, candgen, group=1):
            # ignore stopwords
            if match.get_span().lower() in stopwords:
                continue
            matches[sidx].append(match)

        # build header indices for all sentences
        # this captures what section header a sentence lives under
        curr = [None]
        header_index = {}
        for j in range(len(document.sentences)):
            if j in matches:
                curr = matches[j]
            header_index[j] = curr
        
        for sidx in header_index:
            document.annotations[sidx].update({'HEADER':header_index[sidx]})



###############################################################################
#
# Historical Tagger
#
###############################################################################


ABSTAIN = 0
POSITIVE = 1
NEGATIVE = 2

def LF_underspecified_date(span):
    """
    The DATETIME markup layer only contains datetimes that are easy to normalize with
    high accuracy at the day-level (e.g., January 11, 2000, 11-13-1980). Many notes contain
    additional, but underspecificed dates, e.g.,

    - h/o joint replacements in B hips and shoulders from osteonecrosis in 1980
    - LLE DVT in 2000 and LV thrombus discovered in 2001 during admission for MI

    When we have a note-level timestamp, we can heuristically normalize underspecified dates.

    """
    doc = span.sentence.document
    doc_ts = doc.props['doctime']
    annotations = doc.annotations[span.sentence.position]
    dates = annotations['DATETIME'] if 'DATETIME' in annotations else []

    if not dates or not doc_ts:
        return ABSTAIN

    # year
    for match in re.finditer("(19[0-9]{2}|20[01][0-9])+s*", span.sentence.text, re.I):
        year = int(match.group().strip("s"))
        if doc_ts.year > year:
            return POSITIVE

    # month/date
    m = re.search(r'''on ((1[12]|[1-9])[/-](3[01]|[12][0-9]|[1-9]))\b''', span.sentence.text, re.I)
    if m:
        month, date = map(int, re.split("[/-]", m.group(1)))
        ts = datetime.datetime(doc_ts.year, month, date)
        if doc_ts > ts:
            return POSITIVE

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


def LF_in_history_of_list(span):
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
        for name in ["DISORDER", "PROCEDURE"]:
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

    NOTE: We currently use a more restrictive definition of historical that PyConText,
    labeling any event as historical that only place a event in the past. For example,
    events that are ongoing (i.e., imply past occurences as well) are not labeled as historical

    - chronic pain
    - recurrent infections
    - increase in X
    etc

    """

    def __init__(self, targets, label_reduction='or'):
        self.prop_name = 'hist'
        self.targets = targets
        self.label_reduction = label_reduction

        self.lfs = [
            LF_doctime_complication,
            LF_in_history_headers,
            LF_history_of,
            LF_underspecified_date,
            LF_in_history_of_list
        ]

    def _apply_lfs(self, span):
        """ Apply labeling functions. """
        return np.array([lf(span) for lf in self.lfs])

    def tag(self, document):
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
                        span.props[self.prop_name] = int(1 in L)
                    # label matrix
                    elif L.any() and self.label_reduction == 'matrix':
                        span.props[self.prop_name] = L


class ParentSectionTagger(Tagger):
    
    def __init__(self, targets):
        self.prop_name = 'section'
        self.targets = targets
   
    def tag(self, document):
        for i in document.annotations:
            for layer in self.targets:
                if layer not in document.annotations[i]:
                    continue
                for span in document.annotations[i][layer]:
                    header = span.sentence.document.annotations[span.sentence.i]['HEADER'][-1]
                    span.props[self.prop_name] = header

