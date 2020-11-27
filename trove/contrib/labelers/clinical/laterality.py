from .helpers import *
from inkfish.labelers.taggers import *

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
            'L': ['left', 'lt', 'l', 'left-sided', 'left sided',
                  'l-sided', 'l sided'],
            'R': ['right', 'rt', 'r', 'right-sided', 'right sided',
                  'r-sided', 'rt sided'],
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
                return Span(char_start=span.char_start + start,
                            char_end=span.char_start + end - 1,
                            sentence=sent)

        # check context windows
        left = get_left_span(span, sent, window=window)
        right = get_right_span(span, sent, window=window)

        # left window
        matches = []
        for match in re.finditer(laterality_rgx, left.get_span(), re.I):
            start, end = match.span()
            ts = Span(char_start=left.char_start + start,
                      char_end=left.char_start + end - 1,
                      sentence=sent)
            dist = span.char_start - ts.char_end
            matches.append((dist, ts))
        # return closest match
        if matches:
            for dist, tspan in sorted(matches, reverse=0):
                return tspan

        return None

    def tag(self, document, ngrams=2):
        for i in document.annotations:
            # apply to the following concept targets
            for layer in self.targets:
                if layer not in document.annotations[i]:
                    continue
                for span in document.annotations[i][layer]:
                    laterality = self._get_laterality(span,
                                                      document.sentences[i],
                                                      window=ngrams)
                    if laterality:
                        span.props['laterality'] = \
                            self._get_normed_laterality(laterality)
