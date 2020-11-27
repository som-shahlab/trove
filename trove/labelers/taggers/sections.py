import re
from .taggers import *


###############################################################################
#
# Section Header Taggers
#
###############################################################################

class SectionHeaderTagger(Tagger):
    """
    Identify possible section headers such as:
        HPI:
        HOSPITAL COURSE:
    """
    def __init__(self,
                 header_dict=None,
                 stop_headers=None,
                 max_token_len=6):

        self.stop_headers = {} if not stop_headers else stop_headers
        self.header_dict = {} if not header_dict else {'headers':header_dict}
        self.max_token_len = max_token_len
        if self.header_dict:
            max_ngrams = max([len(t) for t in self.header_dict])
            self.candgen = Ngrams(n_max=max_ngrams, split_on=None)
        self._init()

    def _init(self):
        """
        Regular expression for detecting section headers.
        """
        rgx = '(?:^[\s\n]*|[\n])((?:(?:[A-Za-z#.,]+|24hrs)\s{0,1}){1,' + \
              str(self.max_token_len) + '}[:])'
        self.matchers = {"HEADER": re.compile(rgx, re.I)}

    def _matches(self, rgx, doc, ngrams, group=0):
        """
        For each sentence, return all matches for the provided regex pattern.
        """
        text = ''
        for i, sent in enumerate(doc.sentences):
            matches = set()
            # find all dictionary header matches
            if self.header_dict:
                m = dict_matcher(sent,
                                 self.candgen,
                                 self.header_dict,
                                 min_length=2,
                                 stopwords={})
                if m:
                    matches.update(m['headers'])

            # find all regex matches
            for match in rgx.finditer(sent.text):
                span = match.span(group)
                start, end = span
                # remove trailing colon
                if match.group()[-1] == ':':
                    end -= 1
                tspan = Span(char_start=start, char_end=end - 1, sentence=sent)
                matches.add(tspan)

            for span in matches:
                # filter out some headers
                if span.text in self.stop_headers or '\n' in span.text:
                    continue
                yield (i, span)

    def tag(self, document, ngrams=6, stopwords=[]):
        """ """
        candgen = Ngrams(n_max=ngrams)
        matches = defaultdict(set)
        for sidx, match in self._matches(self.matchers["HEADER"],
                                         document,
                                         candgen,
                                         group=1):
            # ignore stopwords
            if match.get_span().lower() in stopwords:
                continue
            matches[sidx].add(match)

        # sort all by char offset and remove nested spans
        for sidx in matches:
            matches[sidx] = longest_matches(list(matches[sidx]))
            matches[sidx] = sorted(matches[sidx],
                                   key=lambda x:x.abs_char_start,
                                   reverse=0)

        # build header index for all sentences
        curr = [None]
        header_index = {}
        for j in range(len(document.sentences)):
            if j in matches:
                curr = matches[j]
            header_index[j] = curr

        for sidx in header_index:
            document.annotations[sidx].update({'HEADER': header_index[sidx]})


class ParentSectionTagger(Tagger):

    def __init__(self, targets, major_headers=None):
        self.prop_name = 'section'
        self.targets = targets
        self.major_headers = {} if not major_headers else major_headers

    def tag(self, document, **kwargs):
        for i in document.annotations:
            for layer in self.targets:
                if layer not in document.annotations[i]:
                    continue
                # assign all spans to a parent
                for span in document.annotations[i][layer]:
                    # walk up each sentence to find the major section header
                    for j in range(i, -1, -1):
                        # TODO - check all headers found in a sentence
                        h = document.annotations[j]['HEADER'][0]
                        # just assign the closest header tag
                        if not self.major_headers:
                            break
                        elif h and self.major_headers:
                            if h.text in self.major_headers and \
                                    span.abs_char_start > h.abs_char_end:
                                break
                    span.props[self.prop_name] = h
