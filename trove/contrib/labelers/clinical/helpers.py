import re
from trove.dataloaders.contexts import Sentence, Span


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
    try:
        start, end = sentence.char_offsets[i], sentence.char_offsets[j-1] + len(sentence.words[j-1]) - 1
        return Span(char_start=start, char_end=end, sentence=sentence)
    except:
        print('---')
        print(span.sentence.document.name)
        print(span)
        print(window, j, i, sentence.char_offsets, sentence)
        print('---')
        return Span(char_start=0, char_end=-1, sentence=sentence)


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
