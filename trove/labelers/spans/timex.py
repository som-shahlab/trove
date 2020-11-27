import re
from ..data.dataloaders.contexts import Span
from trove.labelers.taggers import Tagger
from collections import defaultdict
from datetime import *
import pdb

#################################################################################
#
#  Sub-patterns
#
#################################################################################

meridiems = r'''([APap][m]|[APap][.][m][.])'''  # AM | PM
clock_time = r'''([0-2]?[0-9][:][0-5][0-9])(:[0-5][0-9])*'''  # 10:00 | 10:10:05

year_range = r'''(19[0-9][0-9]|20[012][0-9])'''  # 1900 - 2020
day_range = r'''(3[01]|[12][0-9]|[0]*[1-9])'''  # 01 - 31
month_range = r'''([1][012]|[0]*[1-9])'''  # 01 - 12

month_full = r'''(january|february|march|april|may|june|july|august|september|october|november|december)'''
month_abbrv = r'''((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[.]*)'''
day_full = r'''(monday|tuesday|wednesday|thursday|friday|saturday|sunday)'''
day_abbrv = r'''(mon|tue|wed|thu|fri|sat|sun)[.]*'''

##################################################################################
#
# Labeling Functions
#
##################################################################################

rgx_time = r'''({clock_time})(\s*{meridiems})*'''.format(clock_time=clock_time, meridiems=meridiems)
rgx_month = r'''({month_full}|{month_abbrv})'''.format(month_full=month_full, month_abbrv=month_abbrv)
rgx_day = r'''({day_full})'''.format(day_full=day_full, day_abbrv=day_abbrv)

# Dates with numeric months
d_params = {'M': month_range, 'D': day_range, 'Y': year_range, 'T': rgx_time}

rgx_y = r'''\b({Y})\b'''.format(**d_params)  # 2019
rgx_mdy = r'''({M}[-]{D}[-]{Y}|{M}[/]{D}[/]{Y})(\s*{T})*'''.format(**d_params)  # 1/11/2000 12:30 PM
rgx_ymd = r'''({Y}[/]{M}[/]{D}|{Y}[-]{M}[-]{D})(\s*{T})*'''.format(**d_params)  # 2000-11-12 12:30 PM
rgx_md = r'''\b(([1][0-9]|[0][1-9])[/][0-3][0-9])\b'''  # 11/31 (implied year)
rgx_ys = r'''\b((mid[-])*({Y}|the [2-9]0)[s])\b'''.format(**d_params)  # 1920s | mid-1990s | the 80s
rgx_mdy2 = r'''\b({M}[-]{D}[-]([012][0-9])|{M}[/]{D}[/]([012][0-9]))(\s*{T})*\b'''.format(**d_params)  # 9/21/09

# Dates with string months
s_params = {'M': rgx_month, 'D': day_range, 'Y': year_range, 'T': rgx_time}

rgx_dash_mdy = r'''({M}[-]{D}[-]{Y})(\s*{T})*'''.format(**s_params)  # Apr-03-2011 13:21:30
rgx_m_of_y = r'''({M} of {Y}|{Y} in {M})'''.format(**s_params)  # January of 2018 | 2005 in April
rgx_month_dy = r'''({M}[ ]*{D}(st|nd|rd|th)*[, ]*{Y})'''.format(**s_params)  # July 30, 2019
rgx_concat_mdy = r'''({D}{M}{Y})'''.format(**s_params)  # 30Jan2019
rgx_d_of_my = r'''({D}(st|nd|rd|th)* of {M}(\s{Y})*)'''.format(**s_params)  # 20th of July 2010
rgx_month_d = r'''\b({M} {D}(st|nd|rd|th)*)\b'''.format(**s_params)  # September 16
rgx_d_m_y = r'''{D}(-|\s*){M}(-|\s*){Y}(\s*{T})*'''.format(**s_params) # 13-Apr-2012 17:35:00 | 15 Jun 2006  5:16PM


# Other TIMEX3 expressions
number_0_10 = r'''(zero|one|two|three|four|five|six|seven|eight|nine|ten)'''
number_11_19 = r'''((thir|four|fif|six|seven|eigh|nine)teen)'''
number_20_90 = r'''((twenty|thirty|fourty|fifty|sixty|seventy|eighty|ninty)([-]*{N})*)'''.format(N=number_0_10)

# five years ago
rgx_number_full = r'''{}|{}|{}'''.format(number_0_10, number_11_19, number_20_90)
rgx_timex_ago = r'''\b(([1-9][0-9]*([.][5])*|({})|few|a) ((year|month|week|day|hour)[s]*) ago)\b'''.format(rgx_number_full)

# relative temporal expressions
rgx_day_parts = r'''\b((this) (morning|afternoon|evening)|(yesterday|today|tomorrow|tonight|tonite)[']*[s]*)\b'''
rgx_day_times = r'''\b(now|currently|presently)\b'''
rgx_day_rela = r'''\b((next|last|this) ({W}|week|month|year))\b'''.format(W=day_full)
rgx_recent_now = r'''\b((current|recent)(ly)*|at this (point|time)|now)\b'''
rgx_operative = r'''\b((pre|post|intra)[-]*(operative(ly)*|op))\b'''

# patterns requiring lookbehinds
prefixes = {'on', 'until', 'before', 'after', 'during', 'started', 'begins'}
lookbehinds = '|'.join(map(lambda x: f'(?<={x}\s)', prefixes))
rgx_look_md = f'(?:{lookbehinds})' + r'''([01][0-9][-/]([012][0-9]|[3][01]))'''  # on 05-29

regexes = [
    '\b{}\b'.format(rgx_time),
    '\b{}\b'.format(rgx_month),
    '\b{}\b'.format(rgx_day),

    rgx_y,
    rgx_mdy,
    rgx_ymd,
    rgx_md,
    rgx_ys,
    rgx_mdy2,

    rgx_dash_mdy,
    rgx_m_of_y,
    rgx_month_dy,
    rgx_concat_mdy,
    rgx_d_of_my,
    rgx_month_d,
    rgx_d_m_y,

    rgx_timex_ago,
    rgx_day_parts,
    rgx_day_rela,

    rgx_recent_now,
    rgx_operative,

    rgx_month
]


#################################################################################
#
#  TIMEX3 Tagger
#
#################################################################################

class Timex3Tagger(Tagger):
    """
    Regular expressions for common datetime patterns.
    - Supports many explicit datetime patterns
    - Supports (some) relative TIMEX3 expressions (X weeks ago)

    """

    def __init__(self, normalizer=None, tag_name='TIMEX3'):
        self.normalizer = normalizer
        self.tag_name = tag_name
        self._init()

    def _matches(self, matchers, doc, ngrams, group=0):
        """For each sentence, return all matches for the provided regex pattern."""
        matches = {}
        for i, sent in enumerate(doc.sentences):
            matches[i] = {}
            for j, rgx in enumerate(matchers):
                for match in re.finditer(rgx, sent.text, re.I):
                    span = match.span(group)
                    start, end = span

                    tspan = Span(char_start=start, char_end=end - 1, sentence=sent)
                    matches[i][(start, end - 1, end - 1 - start)] = tspan

            # return longest, non-overlapping matches
            mask = {}
            for key in sorted(matches[i], key=lambda x: x[-1], reverse=1):
                is_longest = True
                start, end, length = key
                for j in range(start, end):
                    if j not in mask:
                        mask[j] = tspan
                    else:
                        is_longest = False
                tspan = matches[i][key]
                if is_longest:
                    yield (i, tspan)

    def tag(self, document, ngrams=6, stopwords=[]):
        """ """
        matches = defaultdict(list)
        for sidx, match in self._matches(self.matchers[self.tag_name], document, None, group=0):
            if match.get_span().lower() in stopwords:
                continue
            matches[sidx].append(match)

        if self.normalizer:
            self.normalizer.normalize(matches)

        for sidx in matches:
            document.annotations[sidx].update({self.tag_name: matches[sidx]})

    def _init(self):
        """Common datetime regular expressions."""
        self.matchers = {self.tag_name: regexes}


class Timex3NormalizerTagger(Tagger):
    """
    HACK: Wrap existing normalize class to support time normalizations that require doc times a priori.

    Given TIMEX3 strings, normalize to Python datetime objects where possible.
    The Document `doctime` property is used to anchor all date math.

    """

    def __init__(self):
        self.normalizer = TimexNormalizer()

        self.regexes = [
            (rgx_month_d, self.norm_month_d),
            # (rgx_timex_ago, self.norm_x_ago)
            # (rgx_day_parts, self.norm_recent),
            # (rgx_recent_now, self.norm_recent),
            # (rgx_day_times, self.norm_recent)

        ]

    def norm_x_ago(self, span):

        nums = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

        try:
            doctime = span.sentence.document.props['doctime']
            if not doctime:
                return None

            text = span.text.lower()
            rgx = r'''\b(([1-9][0-9]|{}|few|a) ((year|month|week|day|hour)[s]*) ago)\b'''.format(rgx_number_full)

            m = re.search(rgx, s, re.I)
            mult = m.group(2)
            unit = m.group(10)

            try:
                mult = int(mult)
            except:
                if mult in nums:
                    multi = nums[mult]
                else:
                    return None

            print(m)
            print(mult, unit)
            print("--")

        except Exception as e:
            print("norm_x_ago:date normalization error", e, span)
        return None

    def norm_recent(self, span):
        try:
            # HACK
            if re.search(r'''(current|recent)''', span.text, re.I):
                return None

            doctime = span.sentence.document.props['doctime']
            if not doctime:
                return None

            if re.search(r'''(yesterday)''', span.text.lower(), re.I):
                tdelta = timedelta(days=1)
                return doctime - tdelta
            elif re.search(r'''(tomorrow)''', span.text.lower(), re.I):
                tdelta = timedelta(days=1)
                return doctime + tdelta
            return doctime

        except Exception as e:
            print("norm_month_d::date normalization error", e, span)
        return None

    def norm_month_d(self, span):
        """ September 16 """
        try:
            doctime = span.sentence.document.props['doctime']
            if not doctime:
                return None

            month, day = span.text.split()
            month = month.strip(".").lower()
            day = int(day.lower().strip('thndrdst'))

            if month in TimexNormalizer.MONTH_TO_INT:
                # We assign year assuming the closest year based on doctime.
                # For example, if the doctime is Feb. 2010 and the doc mentions "December 19",
                # assume the year is 2009.
                # But if the mention is March 19, we assume 2010. The idea is that at some point
                # a date without a year become ambiguous to a reader.

                year = doctime.year
                month = TimexNormalizer.MONTH_TO_INT[month]

                ts1 = datetime(year, month, day)
                ts2 = datetime(year - 1, month, day)
                alt = ts1 if abs((doctime - ts1).days) < abs((doctime - ts2).days) else ts2

                ts = datetime(year, month, day)
                if ts != alt:
                    print('FLIPPED')
                    print(doctime)
                    print(ts1)
                    print(ts2)
                    print('---')
                    ts = alt

                return ts

        except Exception as e:
            print("norm_month_d::date normalization error", e, span)
        return None

    def tag(self, document, ngrams=None):

        entities = {i: document.annotations[i]['TIMEX3'] for i in document.annotations \
                    if 'TIMEX3' in document.annotations[i]}
        self.normalizer.normalize(entities)

        for i in entities:
            unf = [span for span in entities[i] if span.normalized is None]
            for span in unf:
                for rgx, normf in self.regexes:
                    if re.search(rgx, span.text, re.I):
                        span.normalized = normf(span)
                        break
                # if not span.normalized:
                #    print(span)


class TimexTagger(Tagger):
    """
    Regular expressions for common datetime patterns.

    TODOs:
    - Support common TIMEX expressions for non-numeric relative datetimes, e.g., 'last week', '1 month ago'

    """

    def __init__(self, normalizer=None):
        self.normalizer = normalizer
        self._init()

    def _matches(self, rgx, doc, ngrams, group=0):
        """
        For each sentence, return all matches for the provided regex pattern.
        """
        matches = {}

        debug = set()
        for i, sent in enumerate(doc.sentences):

            # if '[end section' not in sent.text:
            #    continue
            # print(doc.name, [sent.text])

            matches[i] = {}
            for match in re.finditer(rgx, sent.text, re.I):
                span = match.span(group)
                start, end = span

                m = re.search(r"""['s().,!?]$""", match.group(), re.I)
                if m:
                    end -= len(m.group())

                # debug.add(sent.text)

                tspan = Span(char_start=start, char_end=end - 1, sentence=sent)
                matches[i][(start, end - 1, end - 1 - start)] = tspan

            # return longest, non-overlapping matches
            mask = {}
            for key in sorted(matches[i], key=lambda x: x[-1], reverse=1):
                is_longest = True
                start, end, length = key
                for j in range(start, end):
                    if j not in mask:
                        mask[j] = tspan
                    else:
                        is_longest = False
                tspan = matches[i][key]
                if is_longest:
                    print(tspan.text)
                    yield (i, tspan)
                else:
                    print("ERROR overlap", tspan)

        # if debug:
        #    print(debug)

    def tag(self, document, ngrams=6, stopwords=[]):
        """ """
        matches = defaultdict(list)
        for sidx, match in self._matches(self.matchers["DATETIME"], document, None, group=0):
            if match.get_span().lower() in stopwords:
                continue
            matches[sidx].append(match)

        if self.normalizer:
            self.normalizer.normalize(matches)

        for sidx in matches:
            document.annotations[sidx].update({'DATETIME': matches[sidx]})

    def _init(self):
        """
        Common datetime regular expressions.
        """
        time_rgx = r'''[0-2][0-9][:]([0-5][0-9])(\s*([ap]m|[apAP][.]*[mM][.]*))*'''  # 12:30 PM
        datetime_rgx = r'''\d{1,2}/\d{1,2}/(19[0-9]{2}|20[01][0-9])(\s(\d{4}|\d{1,2}[:]\d{2}))*(\s*([ap]m|[apAP][.]*[mM][.]*))*'''  # 11/13/1980 12:30 PM
        month_rgx = r'''\b((january|february|march|april|may|june|july|august|september|october|november|december)|(jan|feb|mar|apr|may|june|jul|aug|sep|sept|oct|nov|dec)[.]*)\b'''
        month_date_year_rgx = month_rgx + r'''\s+(3[01]|[12][0-9]|[1-9])[, ]+(19[0-9]{2}|20[01][0-9])'''
        month_year_rgx = month_rgx + r'''[, ]*(19[0-9]{2}|20[01][0-9])'''

        date1_rgx = r'''((19|20)[0-9]{2}|[0-3][0-9])[-][0-1]{0,1}[0-9][-]([3][01]|[012][0-9]|[0-9])'''  # 2000-11-12
        date2_rgx = r'''\b[0-9]{2}[/][0-9]{2}[/][0-9]{2}\b'''
        date3_rgx = r'''([01][0-9]/[0-3][0-9])(?:\s)'''  # 11/13
        date4_rgx = r'''[0-1]{0,1}[0-9][/]([3][01]|[12][0-9]|[0-9])[/]((19|20)[0-9]{2}|[0-3][0-9])\b'''  # 1/11/2000
        date5_rgx = r'''[0-1]{0,1}[0-9][-]([3][01]|[12][0-9]|[0-9])[-]((19|20)[0-9]{2}|[0-3][0-9])\b'''  # 1/11/2000
        # Added on July 30, 2019 for THYME corpus
        date6_rgx = r'''[0123][0-9][-]*(jan|feb|mar|apr|may|june|jul|aug|sep|sept|oct|nov|dec)[-]*(20[012][0-9]|19[0-9][0-9])'''

        year_rgx = r'''[(]*(([0][1-9]|[1][012]|[1-9])[/])*(20[01][0-9]|19[56789][0-9])[).!,s]*'''
        merge_date_rgx = r'''((31|30)|[12][0-9]|[1-9])(Jan|Feb|Mar|Apr|May|Jun|June|July|Jul|Aug|Sept|Sep|Oct|Nov|Dec)(19|20)*[0-9][0-9]'''
        month_of_year = r'''{} of (19|20)[0-9][0-9]'''.format(month_rgx)

        rgx = "|".join([datetime_rgx, time_rgx, month_date_year_rgx, month_year_rgx,
                        date1_rgx,
                        date2_rgx, date3_rgx, date4_rgx, date5_rgx, year_rgx, date6_rgx,
                        merge_date_rgx, month_of_year])

        self.matchers = {"DATETIME": rgx}


class TimexNormalizer(object):
    """
    TODO: Refactor! This class is messy

    Functionality consists of defining tuples of pattern/normalization pairs. These take the form:
        (regex, normalization function)
    which maps a string to a Python datetime object, e.g.,:
        '2001-1-1' -> datetime.datetime(2001, 1, 1, 0, 0))

    """
    MONTHS_FULL = 'january|february|march|april|may|june|july|august|september|october|november|december'
    MONTHS_ABBRV = 'jan|feb|mar|apr|may|jun|jul|aug|sept|oct|nov|dec'

    MONTH_TO_INT = {month: i + 1 for i, month in enumerate(MONTHS_FULL.split("|"))}
    MONTH_TO_INT.update({month: i + 1 for i, month in enumerate(MONTHS_ABBRV.split("|"))})
    MONTH_TO_INT['sep'] = 9
    MONTHS_ABBRV = MONTHS_ABBRV.replace('sept|', 'sept|sep|')

    YEARS = r'''20[01][0-9]|19[2-9][0-9]'''
    MONTHS = ""
    DATES = ""

    def __init__(self):

        self.norm_map = {}

        # 1: Source Patterns
        date_rgx = r'''\b[0-9]{1,2}[-/][0-9]{1,2}([-/]([0-9]{4}|[0-9]{2}))\b'''
        month_rgx = f'''(({TimexNormalizer.MONTHS_FULL})|({TimexNormalizer.MONTHS_ABBRV})[.]*)'''
        month_date_year_rgx = month_rgx + r'''\s+(3[01]|[12][0-9]|[1-9])[, ]+(19[0-9]{2}|20[01][0-9])'''
        month_year_rgx = month_rgx + r'''[, ]*(19[0-9]{2}|20[01][0-9])'''
        month_of_year = r'''{} of (19|20)[0-9][0-9]'''.format(month_rgx)
        merge_date_rgx = f'''((31|30)|[12][0-9]|[1-9])({TimexNormalizer.MONTHS_ABBRV})(19|20)*[0-9][0-9]'''
        d_month_year_rgx = r'''([0]*[1-9]|[1][012])[/-](20[01][0-9]|19[2-9][0-9])'''
        year_rgx = f'''({TimexNormalizer.YEARS})(?![\-])'''
        year_month_date = r'''((19|20)[0-9]{2}|[0-3][0-9])[-][0-1]{0,1}[0-9][-]([3][01]|[012][0-9]|[0-9])'''
        # THYME patterns
        concat_date = r'''[0123][0-9][-]*(jan|feb|mar|apr|may|june|jul|aug|sep|sept|oct|nov|dec)[-]*(20[012][0-9]|19[0-9][0-9])'''

        # 2: Normalization Map
        norm_mapping = [

            (concat_date, self.date_norm_8),  # 05-Oct-2010 or 30Sep2010

            (date_rgx, self.date_norm_1),  # 7/10/2000 | 7-10-2000
            (month_date_year_rgx, self.date_norm_2),  # January 11, 2000
            (month_year_rgx, self.date_norm_3),  # Jan 2009
            (month_of_year, self.date_norm_4),  # January of 2008
            (merge_date_rgx, self.date_norm_5),  # 03June11
            (year_rgx, self.date_norm_7),  # 2009
            (d_month_year_rgx, self.date_norm_1),  # 01/2009
            (year_month_date, self.date_norm_6)  # 2010-11-12

        ]
        self.norm_map = dict(norm_mapping)

    def date_norm_8(self, m, min_year=1910, max_year=2020):
        try:
            abbrvs = TimexNormalizer.MONTHS_ABBRV.lower()
            match = re.search(f'([0123][0-9])[-]*({abbrvs})[-]*(20[012][0-9]|19[0-9][0-9])', m.group(), re.I)

            day = match.group(1)
            month = match.group(2).lower()
            year = match.group(3)

            if year and month in TimexNormalizer.MONTH_TO_INT:
                day = int(day)
                year = int(year)
                month = TimexNormalizer.MONTH_TO_INT[month]
                return datetime(year, month, day)
        except Exception as e:
            print("date_norm_8::date normalization error", e, m)

        return None

    def date_norm_7(self, m, min_year=1910, max_year=2020):
        try:
            year = int(m.group().strip())
            if year < min_year or year > max_year:
                return None
            return datetime(year, 1, 1)
        except:
            return None

    def date_norm_6(self, m, min_year=1910, max_year=2020):
        try:
            args = re.split("[-/]", m.group().strip("'s().,!?").lower())
            year, month, date = map(int, args)
            if year < min_year or year > max_year:
                return None
            return datetime(year, month, date)
        except:
            return None

    def date_norm_1(self, m, min_year=1910, max_year=2020):

        args = re.split("[-/]", m.group().strip("'s().,!?").lower())

        # for dates of the form 11/13/06, add implied '20' to year
        year = int(args[-1])
        if len(args) == 3 and len(args[-1]) == 2:
            args[-1] = '20' + args[-1] if year >= 0 and year <= 18 else '19' + args[-1]

        # 7/10/2000
        if len(args) == 3:
            month, date, year = list(map(int, args))
            if year < min_year or (month > 12 or month <= 0) or (date > 31 or date <= 0):
                return None
            try:
                return datetime(year, month, date)
            except Exception as e:
                return None

        # 7/2000
        elif len(args) == 2:
            args = list(map(int, args))
            month = args[0] if args[0] < args[1] else args[1]
            year = args[0] if month == args[1] else args[1]
            month, year = int(month), int(year)
            if month > 12 or month < 1 or year < min_year or year > max_year:
                return None
            return datetime(month=month, day=1, year=year)

        # 2009
        elif len(args) == 1:
            month, year = 1, args[0].strip("'s")
            month, year = int(month), int(year)
            if month > 12 or month < 1 or year < min_year or year > max_year:
                return None
            return datetime(month=month, day=1, year=year)

        else:
            print("Unrecognized format", s)

        return None

    def date_norm_2(self, m):
        # January 11, 2000
        try:
            values = m.group().strip().lower().replace(",", " ").split()
            year = int(values[2])
            date = int(values[1])
            month = TimexNormalizer.MONTH_TO_INT[values[0]]
            return datetime(year, month, date)
        except Exception as e:
            print("date_norm_2::date normalization error", e, m)
        return None

    def date_norm_3(self, m):
        # Jan 2009
        # December,2008
        try:
            s = m.group().strip().lower()
            s = re.sub("\s{2,}", " ", s)
            s = s.replace(",", " ")
            month, year = s.split()
            month = month.strip(".")

            if year and month in TimexNormalizer.MONTH_TO_INT:
                year = int(year)
                month = TimexNormalizer.MONTH_TO_INT[month]
                return datetime(year, month, 1)
        except Exception as e:
            print("date_norm_3::date normalization error", e, m)
        return None

    def date_norm_4(self, m):
        # January of 2008
        t = m.group().lower().replace(" of ", " ")
        try:
            month, year = t.split()
            month = month.strip(".")
            if year and month in TimexNormalizer.MONTH_TO_INT:
                year = int(year)
                month = TimexNormalizer.MONTH_TO_INT[month]
                return datetime(year, month, 1)
        except Exception as e:
            print("date_norm_4::date normalization error", e, m)
        return None

    def date_norm_5(self, m):
        # 03June11
        t = m.group().lower()
        m = re.search(r'''([0-9]+)([A-Za-z]+)([0-9]+)''', t, re.I)
        if not m:
            return None
        day, month, year = m.group(1), m.group(2), m.group(3)
        if len(year) == 2:
            year = '20' + year
        if month not in TimexNormalizer.MONTH_TO_INT:
            print("MONTH error", month)
        day = int(day)
        month = TimexNormalizer.MONTH_TO_INT[month]
        year = int(year)
        return datetime(year, month, day)

    def _filter(self, span):
        # Filter out times, e.g., 12:30 PM
        time_rgx = r'''^[0-2][0-9][:]([0-5][0-9])(\s*([ap]m|[apAP][.]*[mM][.]*))*$'''
        t = span.get_span().lower().strip()
        return True if re.search(time_rgx, t, re.I) else False

    def normalize(self, markup):
        # Normalize unambiguous dates
        normed = {}
        for i in markup:
            normed[i] = []
            for timex in markup[i]:
                if self._filter(timex):
                    continue
                ts = self._normalize_timex_str(timex.text)
                # print(timex.text)
                if ts:
                    timex.normalized = ts
                    normed[i].append(timex)
                # print(timex.normalized)
                # print("-"* 50)

        return normed

    def _normalize_timex_str(self, seq):
        for rgx in self.norm_map:
            m = re.search(rgx, seq, re.I)
            if m:
                return self.norm_map[rgx](m)
        return None