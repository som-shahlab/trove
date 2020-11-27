import re
from itertools import product
from collections import defaultdict, namedtuple
from trove.dataloaders.contexts import Span, Relation


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
    Apply secondary tokenization rule, e.g.,
    split on hyphens or forward slashes.
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
    """
    TODO: Refactor to remove need for this function

    :param matches:
    :return:
    """
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
        if (i >= curr.char_start and i <= curr.char_end) and \
                (j >= curr.char_start and j <= curr.char_end):
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
            text = re.sub(r'''\s{2,}|\n+''', ' ', span.text).strip()

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
# Taggers
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
# Dictionary Tagger
#
###############################################################################

class DictionaryTagger(Tagger):

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

        candgen = Ngrams(n_max=ngrams, split_on=self.split_on)
        for sent in document.sentences:
            m = dict_matcher(sent,
                             candgen,
                             self.dictionaries,
                             min_length=self.min_length,
                             stopwords=self.stopwords)

            if m:
                if sent.position not in document.annotations:
                    print('ERROR - sent.position not in doc annotations',
                          sent.position, dict(m))
                    continue
                document.annotations[sent.position].update(dict(m))

###############################################################################
#
# Precomputed Entity Tagger
#
###############################################################################

EntityTag = namedtuple(
    'EntityTag', 'doc_name term abs_char_start abs_char_end'
)

class PrecomputedEntityTagger(Tagger):

    def __init__(self, fpath, type_name):
        self.type_name = type_name
        self.annotations = self._load_annotations(fpath)

    def _load_annotations(self, fpath):
        annos = defaultdict(list)
        col_names = ['doc_name', 'term', 'abs_char_start', 'abs_char_end']
        df = pd.read_csv(fpath,
                         sep='\t',
                         names=col_names)
        for row in df.itertuples():
            entity = EntityTag(row.doc_name, row.term, row.abs_char_start,
                               row.abs_char_end)
            annos[row.doc_name].append(entity)
        print(f"Loaded {len(annos)} document [{self.type_name}] entities")
        return annos

    def _get_span_sentence(self, abs_char_start, abs_char_end, sentences):
        for sent in sentences:
            end_str_len = len(sent.words[-1])
            if abs_char_start >= sent.abs_char_offsets[0] and abs_char_end <= (
                    sent.abs_char_offsets[-1] + end_str_len):
                return sent
        return None

    def _is_overlapping(self, a, b):
        if a.abs_char_start >= b.abs_char_start and a.abs_char_start <= b.abs_char_end:
            return True
        if a.abs_char_end >= b.abs_char_start and a.abs_char_end <= b.abs_char_end:
            return True
        if b.abs_char_start >= a.abs_char_start and b.abs_char_start <= a.abs_char_end:
            return True
        if b.abs_char_end >= a.abs_char_start and b.abs_char_end <= a.abs_char_end:
            return True
        return False

    def tag(self, document, ngrams=None):
        """
        Use existing labeled data to generate Span objects
        """
        if document.name not in self.annotations:
            return

        n_errs = 0
        entities = {sent.i: {} for sent in document.sentences}
        for anno in self.annotations[document.name]:
            # get parent sentence for this span
            sent = self._get_span_sentence(anno.abs_char_start,
                                           anno.abs_char_end,
                                           document.sentences)
            if not sent:
                n_errs += 1
                continue

            offset = sent.abs_char_offsets[0]
            span = Span(anno.abs_char_start - offset,
                        anno.abs_char_end - offset, sentence=sent)

            # HACK -- exclude all entities that are overlapping/nested
            # within header spans (TODO move to seprate pipeline module)
            ignore_span = False
            if 'HEADER' in document.annotations[sent.i]:
                for h in document.annotations[sent.i]['HEADER']:
                    if h is not None and self._is_overlapping(h, span):
                        ignore_span = True
                        break

            if ignore_span:
                continue

            if self.type_name not in entities[sent.i]:
                entities[sent.i][self.type_name] = []
            entities[sent.i][self.type_name].append(span)

        for i in entities:
            document.annotations[i].update(entities[i])

        if n_errs > 0:
            print(f'Skipped {document.name}({n_errs}) entities')

###############################################################################
#
# Relation Tagger
#
###############################################################################

class RelationTagger(object):

    def __init__(self, type_name, arg_types):
        self.type_name = type_name
        self.arg_types = arg_types

    def tag(self, document, **kwargs):
        for i in document.annotations:
            # skip sentence when all argument types are not present
            if len(set(document.annotations[i]).intersection(
                    self.arg_types)) != len(self.arg_types):
                continue

            # relations as Cartesian product of all typed argument spans
            args = [(name, document.annotations[i][name]) for name in
                    self.arg_types]
            args = list(product(*[spans[1] for spans in args]))
            relations = [
                Relation(self.type_name, args=dict(zip(self.arg_types, rela)))
                for rela in args
            ]
            document.annotations[i].update({self.type_name: relations})
